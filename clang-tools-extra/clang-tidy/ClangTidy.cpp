//===--- tools/extra/clang-tidy/ClangTidy.cpp - Clang tidy tool -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
///  \file This file implements a clang-tidy tool.
///
///  This tool uses the Clang Tooling infrastructure, see
///    http://clang.llvm.org/docs/HowToSetupToolingForLLVM.html
///  for details on setting it up with LLVM source tree.
///
//===----------------------------------------------------------------------===//

#include "ClangTidy.h"
#include "ClangTidyDiagnosticConsumer.h"
#include "ClangTidyModuleRegistry.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Rewrite/Frontend/FixItRewriter.h"
#include "clang/Rewrite/Frontend/FrontendActions.h"
#include "clang/StaticAnalyzer/Core/BugReporter/PathDiagnostic.h"
#include "clang/StaticAnalyzer/Frontend/AnalysisConsumer.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/ReplacementsYaml.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Signals.h"
#include <algorithm>
#include <utility>

using namespace clang::ast_matchers;
using namespace clang::driver;
using namespace clang::tooling;
using namespace llvm;

LLVM_INSTANTIATE_REGISTRY(clang::tidy::ClangTidyModuleRegistry)

namespace clang {
namespace tidy {

namespace {
static const char *AnalyzerCheckNamePrefix = "clang-analyzer-";

static const StringRef StaticAnalyzerChecks[] = {
#define GET_CHECKERS
#define CHECKER(FULLNAME, CLASS, DESCFILE, HELPTEXT, GROUPINDEX, HIDDEN)       \
  FULLNAME,
#include "clang/StaticAnalyzer/Checkers/Checkers.inc"
#undef CHECKER
#undef GET_CHECKERS
};

class AnalyzerDiagnosticConsumer : public ento::PathDiagnosticConsumer {
public:
  AnalyzerDiagnosticConsumer(ClangTidyContext &Context) : Context(Context) {}

  void FlushDiagnosticsImpl(std::vector<const ento::PathDiagnostic *> &Diags,
                            FilesMade *filesMade) override {
    for (const ento::PathDiagnostic *PD : Diags) {
      SmallString<64> CheckName(AnalyzerCheckNamePrefix);
      CheckName += PD->getCheckName();
      Context.diag(CheckName, PD->getLocation().asLocation(),
                   PD->getShortDescription())
          << PD->path.back()->getRanges();

      for (const auto &DiagPiece :
           PD->path.flatten(/*ShouldFlattenMacros=*/true)) {
        Context.diag(CheckName, DiagPiece->getLocation().asLocation(),
                     DiagPiece->getString(), DiagnosticIDs::Note)
            << DiagPiece->getRanges();
      }
    }
  }

  StringRef getName() const override { return "ClangTidyDiags"; }
  bool supportsLogicalOpControlFlow() const override { return true; }
  bool supportsCrossFileDiagnostics() const override { return true; }

private:
  ClangTidyContext &Context;
};

class ErrorReporter {
public:
  ErrorReporter(bool ApplyFixes)
      : Files(FileSystemOptions()), DiagOpts(new DiagnosticOptions()),
        DiagPrinter(new TextDiagnosticPrinter(llvm::outs(), &*DiagOpts)),
        Diags(IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs), &*DiagOpts,
              DiagPrinter),
        SourceMgr(Diags, Files), Rewrite(SourceMgr, LangOpts),
        ApplyFixes(ApplyFixes), TotalFixes(0), AppliedFixes(0),
        WarningsAsErrors(0) {
    DiagOpts->ShowColors = llvm::sys::Process::StandardOutHasColors();
    DiagPrinter->BeginSourceFile(LangOpts);
  }

  SourceManager &getSourceManager() { return SourceMgr; }

  void reportDiagnostic(const ClangTidyError &Error) {
    const ClangTidyMessage &Message = Error.Message;
    SourceLocation Loc = getLocation(Message.FilePath, Message.FileOffset);
    // Contains a pair for each attempted fix: location and whether the fix was
    // applied successfully.
    SmallVector<std::pair<SourceLocation, bool>, 4> FixLocations;
    {
      auto Level = static_cast<DiagnosticsEngine::Level>(Error.DiagLevel);
      std::string Name = Error.CheckName;
      if (Error.IsWarningAsError) {
        Name += ",-warnings-as-errors";
        Level = DiagnosticsEngine::Error;
        WarningsAsErrors++;
      }
      auto Diag = Diags.Report(Loc, Diags.getCustomDiagID(Level, "%0 [%1]"))
                  << Message.Message << Name;
      for (const auto &FileAndReplacements : Error.Fix) {
        for (const auto &Replacement : FileAndReplacements.second) {
          // Retrieve the source range for applicable fixes. Macro definitions
          // on the command line have locations in a virtual buffer and don't
          // have valid file paths and are therefore not applicable.
          SourceRange Range;
          SourceLocation FixLoc;
          if (Replacement.isApplicable()) {
            SmallString<128> FixAbsoluteFilePath = Replacement.getFilePath();
            Files.makeAbsolutePath(FixAbsoluteFilePath);
            FixLoc = getLocation(FixAbsoluteFilePath, Replacement.getOffset());
            SourceLocation FixEndLoc =
                FixLoc.getLocWithOffset(Replacement.getLength());
            Range = SourceRange(FixLoc, FixEndLoc);
            Diag << FixItHint::CreateReplacement(
                Range, Replacement.getReplacementText());
          }

          ++TotalFixes;
          if (ApplyFixes) {
            bool Success =
                Replacement.isApplicable() && Replacement.apply(Rewrite);
            if (Success)
              ++AppliedFixes;
            FixLocations.push_back(std::make_pair(FixLoc, Success));
          }
        }
      }
    }
    for (auto Fix : FixLocations) {
      Diags.Report(Fix.first, Fix.second ? diag::note_fixit_applied
                                         : diag::note_fixit_failed);
    }
    for (const ClangTidyMessage &Note : Error.Notes)
      reportNote(Note);
  }

  void Finish() {
    // FIXME: Run clang-format on changes.
    if (ApplyFixes && TotalFixes > 0) {
      llvm::errs() << "clang-tidy applied " << AppliedFixes << " of "
                   << TotalFixes << " suggested fixes.\n";
      Rewrite.overwriteChangedFiles();
    }
  }

  unsigned getWarningsAsErrorsCount() const { return WarningsAsErrors; }

private:
  SourceLocation getLocation(StringRef FilePath, unsigned Offset) {
    if (FilePath.empty())
      return SourceLocation();

    const FileEntry *File = SourceMgr.getFileManager().getFile(FilePath);
    FileID ID = SourceMgr.createFileID(File, SourceLocation(), SrcMgr::C_User);
    return SourceMgr.getLocForStartOfFile(ID).getLocWithOffset(Offset);
  }

  void reportNote(const ClangTidyMessage &Message) {
    SourceLocation Loc = getLocation(Message.FilePath, Message.FileOffset);
    DiagnosticBuilder Diag =
        Diags.Report(Loc, Diags.getCustomDiagID(DiagnosticsEngine::Note, "%0"))
        << Message.Message;
  }

  FileManager Files;
  LangOptions LangOpts; // FIXME: use langopts from each original file
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts;
  DiagnosticConsumer *DiagPrinter;
  DiagnosticsEngine Diags;
  SourceManager SourceMgr;
  Rewriter Rewrite;
  bool ApplyFixes;
  unsigned TotalFixes;
  unsigned AppliedFixes;
  unsigned WarningsAsErrors;
};

class ClangTidyASTConsumer : public MultiplexConsumer {
public:
  ClangTidyASTConsumer(std::vector<std::unique_ptr<ASTConsumer>> Consumers,
                       std::unique_ptr<ast_matchers::MatchFinder> Finder,
                       std::vector<std::unique_ptr<ClangTidyCheck>> Checks)
      : MultiplexConsumer(std::move(Consumers)), Finder(std::move(Finder)),
        Checks(std::move(Checks)) {}

private:
  std::unique_ptr<ast_matchers::MatchFinder> Finder;
  std::vector<std::unique_ptr<ClangTidyCheck>> Checks;
};

} // namespace

ClangTidyASTConsumerFactory::ClangTidyASTConsumerFactory(
    ClangTidyContext &Context)
    : Context(Context), CheckFactories(new ClangTidyCheckFactories) {
  for (ClangTidyModuleRegistry::iterator I = ClangTidyModuleRegistry::begin(),
                                         E = ClangTidyModuleRegistry::end();
       I != E; ++I) {
    std::unique_ptr<ClangTidyModule> Module(I->instantiate());
    Module->addCheckFactories(*CheckFactories);
  }
}

static void setStaticAnalyzerCheckerOpts(const ClangTidyOptions &Opts,
                                         AnalyzerOptionsRef AnalyzerOptions) {
  StringRef AnalyzerPrefix(AnalyzerCheckNamePrefix);
  for (const auto &Opt : Opts.CheckOptions) {
    StringRef OptName(Opt.first);
    if (!OptName.startswith(AnalyzerPrefix))
      continue;
    AnalyzerOptions->Config[OptName.substr(AnalyzerPrefix.size())] = Opt.second;
  }
}

std::unique_ptr<clang::ASTConsumer>
ClangTidyASTConsumerFactory::CreateASTConsumer(
    clang::CompilerInstance &Compiler, StringRef File) {
  // FIXME: Move this to a separate method, so that CreateASTConsumer doesn't
  // modify Compiler.
  Context.setSourceManager(&Compiler.getSourceManager());
  Context.setCurrentFile(File);
  Context.setASTContext(&Compiler.getASTContext());

  auto WorkingDir = Compiler.getSourceManager()
                        .getFileManager()
                        .getVirtualFileSystem()
                        ->getCurrentWorkingDirectory();
  if (WorkingDir)
    Context.setCurrentBuildDirectory(WorkingDir.get());

  std::vector<std::unique_ptr<ClangTidyCheck>> Checks;
  CheckFactories->createChecks(&Context, Checks);

  ast_matchers::MatchFinder::MatchFinderOptions FinderOptions;
  if (auto *P = Context.getCheckProfileData())
    FinderOptions.CheckProfiling.emplace(P->Records);

  std::unique_ptr<ast_matchers::MatchFinder> Finder(
      new ast_matchers::MatchFinder(std::move(FinderOptions)));

  for (auto &Check : Checks) {
    Check->registerMatchers(&*Finder);
    Check->registerPPCallbacks(Compiler);
  }

  std::vector<std::unique_ptr<ASTConsumer>> Consumers;
  if (!Checks.empty())
    Consumers.push_back(Finder->newASTConsumer());

  AnalyzerOptionsRef AnalyzerOptions = Compiler.getAnalyzerOpts();
  // FIXME: Remove this option once clang's cfg-temporary-dtors option defaults
  // to true.
  AnalyzerOptions->Config["cfg-temporary-dtors"] =
      Context.getOptions().AnalyzeTemporaryDtors ? "true" : "false";

  GlobList &Filter = Context.getChecksFilter();
  AnalyzerOptions->CheckersControlList = getCheckersControlList(Filter);
  if (!AnalyzerOptions->CheckersControlList.empty()) {
    setStaticAnalyzerCheckerOpts(Context.getOptions(), AnalyzerOptions);
    AnalyzerOptions->AnalysisStoreOpt = RegionStoreModel;
    AnalyzerOptions->AnalysisDiagOpt = PD_NONE;
    AnalyzerOptions->AnalyzeNestedBlocks = true;
    AnalyzerOptions->eagerlyAssumeBinOpBifurcation = true;
    std::unique_ptr<ento::AnalysisASTConsumer> AnalysisConsumer =
        ento::CreateAnalysisConsumer(Compiler);
    AnalysisConsumer->AddDiagnosticConsumer(
        new AnalyzerDiagnosticConsumer(Context));
    Consumers.push_back(std::move(AnalysisConsumer));
  }
  return llvm::make_unique<ClangTidyASTConsumer>(
      std::move(Consumers), std::move(Finder), std::move(Checks));
}

std::vector<std::string> ClangTidyASTConsumerFactory::getCheckNames() {
  std::vector<std::string> CheckNames;
  GlobList &Filter = Context.getChecksFilter();
  for (const auto &CheckFactory : *CheckFactories) {
    if (Filter.contains(CheckFactory.first))
      CheckNames.push_back(CheckFactory.first);
  }

  for (const auto &AnalyzerCheck : getCheckersControlList(Filter))
    CheckNames.push_back(AnalyzerCheckNamePrefix + AnalyzerCheck.first);

  std::sort(CheckNames.begin(), CheckNames.end());
  return CheckNames;
}

ClangTidyOptions::OptionMap ClangTidyASTConsumerFactory::getCheckOptions() {
  ClangTidyOptions::OptionMap Options;
  std::vector<std::unique_ptr<ClangTidyCheck>> Checks;
  CheckFactories->createChecks(&Context, Checks);
  for (const auto &Check : Checks)
    Check->storeOptions(Options);
  return Options;
}

ClangTidyASTConsumerFactory::CheckersList
ClangTidyASTConsumerFactory::getCheckersControlList(GlobList &Filter) {
  CheckersList List;

  bool AnalyzerChecksEnabled = false;
  for (StringRef CheckName : StaticAnalyzerChecks) {
    std::string Checker((AnalyzerCheckNamePrefix + CheckName).str());
    AnalyzerChecksEnabled =
        AnalyzerChecksEnabled ||
        (!CheckName.startswith("debug") && Filter.contains(Checker));
  }

  if (AnalyzerChecksEnabled) {
    // Run our regex against all possible static analyzer checkers.  Note that
    // debug checkers print values / run programs to visualize the CFG and are
    // thus not applicable to clang-tidy in general.
    //
    // Always add all core checkers if any other static analyzer checks are
    // enabled. This is currently necessary, as other path sensitive checks
    // rely on the core checkers.
    for (StringRef CheckName : StaticAnalyzerChecks) {
      std::string Checker((AnalyzerCheckNamePrefix + CheckName).str());

      if (CheckName.startswith("core") ||
          (!CheckName.startswith("debug") && Filter.contains(Checker)))
        List.push_back(std::make_pair(CheckName, true));
    }
  }
  return List;
}

DiagnosticBuilder ClangTidyCheck::diag(SourceLocation Loc, StringRef Message,
                                       DiagnosticIDs::Level Level) {
  return Context->diag(CheckName, Loc, Message, Level);
}

void ClangTidyCheck::run(const ast_matchers::MatchFinder::MatchResult &Result) {
  Context->setSourceManager(Result.SourceManager);
  check(Result);
}

OptionsView::OptionsView(StringRef CheckName,
                         const ClangTidyOptions::OptionMap &CheckOptions)
    : NamePrefix(CheckName.str() + "."), CheckOptions(CheckOptions) {}

std::string OptionsView::get(StringRef LocalName, StringRef Default) const {
  const auto &Iter = CheckOptions.find(NamePrefix + LocalName.str());
  if (Iter != CheckOptions.end())
    return Iter->second;
  return Default;
}

std::string OptionsView::getLocalOrGlobal(StringRef LocalName,
                                          StringRef Default) const {
  auto Iter = CheckOptions.find(NamePrefix + LocalName.str());
  if (Iter != CheckOptions.end())
    return Iter->second;
  // Fallback to global setting, if present.
  Iter = CheckOptions.find(LocalName.str());
  if (Iter != CheckOptions.end())
    return Iter->second;
  return Default;
}

void OptionsView::store(ClangTidyOptions::OptionMap &Options,
                        StringRef LocalName, StringRef Value) const {
  Options[NamePrefix + LocalName.str()] = Value;
}

void OptionsView::store(ClangTidyOptions::OptionMap &Options,
                        StringRef LocalName, int64_t Value) const {
  store(Options, LocalName, llvm::itostr(Value));
}

std::vector<std::string> getCheckNames(const ClangTidyOptions &Options) {
  clang::tidy::ClangTidyContext Context(
      llvm::make_unique<DefaultOptionsProvider>(ClangTidyGlobalOptions(),
                                                Options));
  ClangTidyASTConsumerFactory Factory(Context);
  return Factory.getCheckNames();
}

ClangTidyOptions::OptionMap getCheckOptions(const ClangTidyOptions &Options) {
  clang::tidy::ClangTidyContext Context(
      llvm::make_unique<DefaultOptionsProvider>(ClangTidyGlobalOptions(),
                                                Options));
  ClangTidyASTConsumerFactory Factory(Context);
  return Factory.getCheckOptions();
}

ClangTidyStats
runClangTidy(std::unique_ptr<ClangTidyOptionsProvider> OptionsProvider,
             const tooling::CompilationDatabase &Compilations,
             ArrayRef<std::string> InputFiles,
             std::vector<ClangTidyError> *Errors, ProfileData *Profile) {
  ClangTool Tool(Compilations, InputFiles);
  clang::tidy::ClangTidyContext Context(std::move(OptionsProvider));

  // Add extra arguments passed by the clang-tidy command-line.
  ArgumentsAdjuster PerFileExtraArgumentsInserter =
      [&Context](const CommandLineArguments &Args, StringRef Filename) {
        ClangTidyOptions Opts = Context.getOptionsForFile(Filename);
        CommandLineArguments AdjustedArgs = Args;
        if (Opts.ExtraArgsBefore) {
          auto I = AdjustedArgs.begin();
          if (I != AdjustedArgs.end() && !StringRef(*I).startswith("-"))
            ++I; // Skip compiler binary name, if it is there.
          AdjustedArgs.insert(I, Opts.ExtraArgsBefore->begin(),
                              Opts.ExtraArgsBefore->end());
        }
        if (Opts.ExtraArgs)
          AdjustedArgs.insert(AdjustedArgs.end(), Opts.ExtraArgs->begin(),
                              Opts.ExtraArgs->end());
        return AdjustedArgs;
      };

  // Remove plugins arguments.
  ArgumentsAdjuster PluginArgumentsRemover =
      [&Context](const CommandLineArguments &Args, StringRef Filename) {
        CommandLineArguments AdjustedArgs;
        for (size_t I = 0, E = Args.size(); I < E; ++I) {
          if (I + 4 < Args.size() && Args[I] == "-Xclang" &&
              (Args[I + 1] == "-load" || Args[I + 1] == "-add-plugin" ||
               StringRef(Args[I + 1]).startswith("-plugin-arg-")) &&
              Args[I + 2] == "-Xclang") {
            I += 3;
          } else
            AdjustedArgs.push_back(Args[I]);
        }
        return AdjustedArgs;
      };

  Tool.appendArgumentsAdjuster(PerFileExtraArgumentsInserter);
  Tool.appendArgumentsAdjuster(PluginArgumentsRemover);
  if (Profile)
    Context.setCheckProfileData(Profile);

  ClangTidyDiagnosticConsumer DiagConsumer(Context);

  Tool.setDiagnosticConsumer(&DiagConsumer);

  class ActionFactory : public FrontendActionFactory {
  public:
    ActionFactory(ClangTidyContext &Context) : ConsumerFactory(Context) {}
    FrontendAction *create() override { return new Action(&ConsumerFactory); }

  private:
    class Action : public ASTFrontendAction {
    public:
      Action(ClangTidyASTConsumerFactory *Factory) : Factory(Factory) {}
      std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &Compiler,
                                                     StringRef File) override {
        return Factory->CreateASTConsumer(Compiler, File);
      }

    private:
      ClangTidyASTConsumerFactory *Factory;
    };

    ClangTidyASTConsumerFactory ConsumerFactory;
  };

  ActionFactory Factory(Context);
  Tool.run(&Factory);
  *Errors = Context.getErrors();
  return Context.getStats();
}

void handleErrors(const std::vector<ClangTidyError> &Errors, bool Fix,
                  unsigned &WarningsAsErrorsCount) {
  ErrorReporter Reporter(Fix);
  vfs::FileSystem &FileSystem =
      *Reporter.getSourceManager().getFileManager().getVirtualFileSystem();
  auto InitialWorkingDir = FileSystem.getCurrentWorkingDirectory();
  if (!InitialWorkingDir)
    llvm::report_fatal_error("Cannot get current working path.");

  for (const ClangTidyError &Error : Errors) {
    if (!Error.BuildDirectory.empty()) {
      // By default, the working directory of file system is the current
      // clang-tidy running directory.
      //
      // Change the directory to the one used during the analysis.
      FileSystem.setCurrentWorkingDirectory(Error.BuildDirectory);
    }
    Reporter.reportDiagnostic(Error);
    // Return to the initial directory to correctly resolve next Error.
    FileSystem.setCurrentWorkingDirectory(InitialWorkingDir.get());
  }
  Reporter.Finish();
  WarningsAsErrorsCount += Reporter.getWarningsAsErrorsCount();
}

void exportReplacements(const std::vector<ClangTidyError> &Errors,
                        raw_ostream &OS) {
  tooling::TranslationUnitReplacements TUR;
  for (const ClangTidyError &Error : Errors) {
    for (const auto &FileAndFixes : Error.Fix)
      TUR.Replacements.insert(TUR.Replacements.end(),
                              FileAndFixes.second.begin(),
                              FileAndFixes.second.end());
  }

  yaml::Output YAML(OS);
  YAML << TUR;
}

} // namespace tidy
} // namespace clang
