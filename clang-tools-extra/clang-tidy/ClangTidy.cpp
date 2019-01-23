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
#include "ClangTidyProfiling.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Config/config.h"
#include "clang/Format/Format.h"
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
#if CLANG_ENABLE_STATIC_ANALYZER
#include "clang/StaticAnalyzer/Core/BugReporter/PathDiagnostic.h"
#include "clang/StaticAnalyzer/Frontend/AnalysisConsumer.h"
#endif // CLANG_ENABLE_STATIC_ANALYZER
#include "clang/Tooling/DiagnosticsYaml.h"
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
#if CLANG_ENABLE_STATIC_ANALYZER
static const char *AnalyzerCheckNamePrefix = "clang-analyzer-";

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
#endif // CLANG_ENABLE_STATIC_ANALYZER

class ErrorReporter {
public:
  ErrorReporter(ClangTidyContext &Context, bool ApplyFixes,
                llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> BaseFS)
      : Files(FileSystemOptions(), BaseFS), DiagOpts(new DiagnosticOptions()),
        DiagPrinter(new TextDiagnosticPrinter(llvm::outs(), &*DiagOpts)),
        Diags(IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs), &*DiagOpts,
              DiagPrinter),
        SourceMgr(Diags, Files), Context(Context), ApplyFixes(ApplyFixes),
        TotalFixes(0), AppliedFixes(0), WarningsAsErrors(0) {
    DiagOpts->ShowColors = llvm::sys::Process::StandardOutHasColors();
    DiagPrinter->BeginSourceFile(LangOpts);
  }

  SourceManager &getSourceManager() { return SourceMgr; }

  void reportDiagnostic(const ClangTidyError &Error) {
    const tooling::DiagnosticMessage &Message = Error.Message;
    SourceLocation Loc = getLocation(Message.FilePath, Message.FileOffset);
    // Contains a pair for each attempted fix: location and whether the fix was
    // applied successfully.
    SmallVector<std::pair<SourceLocation, bool>, 4> FixLocations;
    {
      auto Level = static_cast<DiagnosticsEngine::Level>(Error.DiagLevel);
      std::string Name = Error.DiagnosticName;
      if (Error.IsWarningAsError) {
        Name += ",-warnings-as-errors";
        Level = DiagnosticsEngine::Error;
        WarningsAsErrors++;
      }
      auto Diag = Diags.Report(Loc, Diags.getCustomDiagID(Level, "%0 [%1]"))
                  << Message.Message << Name;
      for (const auto &FileAndReplacements : Error.Fix) {
        for (const auto &Repl : FileAndReplacements.second) {
          SourceLocation FixLoc;
          ++TotalFixes;
          bool CanBeApplied = false;
          if (Repl.isApplicable()) {
            SmallString<128> FixAbsoluteFilePath = Repl.getFilePath();
            Files.makeAbsolutePath(FixAbsoluteFilePath);
            if (ApplyFixes) {
              tooling::Replacement R(FixAbsoluteFilePath, Repl.getOffset(),
                                     Repl.getLength(),
                                     Repl.getReplacementText());
              Replacements &Replacements = FileReplacements[R.getFilePath()];
              llvm::Error Err = Replacements.add(R);
              if (Err) {
                // FIXME: Implement better conflict handling.
                llvm::errs() << "Trying to resolve conflict: "
                             << llvm::toString(std::move(Err)) << "\n";
                unsigned NewOffset =
                    Replacements.getShiftedCodePosition(R.getOffset());
                unsigned NewLength = Replacements.getShiftedCodePosition(
                                         R.getOffset() + R.getLength()) -
                                     NewOffset;
                if (NewLength == R.getLength()) {
                  R = Replacement(R.getFilePath(), NewOffset, NewLength,
                                  R.getReplacementText());
                  Replacements = Replacements.merge(tooling::Replacements(R));
                  CanBeApplied = true;
                  ++AppliedFixes;
                } else {
                  llvm::errs()
                      << "Can't resolve conflict, skipping the replacement.\n";
                }

              } else {
                CanBeApplied = true;
                ++AppliedFixes;
              }
            }
            FixLoc = getLocation(FixAbsoluteFilePath, Repl.getOffset());
            SourceLocation FixEndLoc =
                FixLoc.getLocWithOffset(Repl.getLength());
            // Retrieve the source range for applicable fixes. Macro definitions
            // on the command line have locations in a virtual buffer and don't
            // have valid file paths and are therefore not applicable.
            CharSourceRange Range =
                CharSourceRange::getCharRange(SourceRange(FixLoc, FixEndLoc));
            Diag << FixItHint::CreateReplacement(Range,
                                                 Repl.getReplacementText());
          }

          if (ApplyFixes)
            FixLocations.push_back(std::make_pair(FixLoc, CanBeApplied));
        }
      }
    }
    for (auto Fix : FixLocations) {
      Diags.Report(Fix.first, Fix.second ? diag::note_fixit_applied
                                         : diag::note_fixit_failed);
    }
    for (const auto &Note : Error.Notes)
      reportNote(Note);
  }

  void Finish() {
    if (ApplyFixes && TotalFixes > 0) {
      Rewriter Rewrite(SourceMgr, LangOpts);
      for (const auto &FileAndReplacements : FileReplacements) {
        StringRef File = FileAndReplacements.first();
        llvm::ErrorOr<std::unique_ptr<MemoryBuffer>> Buffer =
            SourceMgr.getFileManager().getBufferForFile(File);
        if (!Buffer) {
          llvm::errs() << "Can't get buffer for file " << File << ": "
                       << Buffer.getError().message() << "\n";
          // FIXME: Maybe don't apply fixes for other files as well.
          continue;
        }
        StringRef Code = Buffer.get()->getBuffer();
        auto Style = format::getStyle(
            *Context.getOptionsForFile(File).FormatStyle, File, "none");
        if (!Style) {
          llvm::errs() << llvm::toString(Style.takeError()) << "\n";
          continue;
        }
        llvm::Expected<tooling::Replacements> Replacements =
            format::cleanupAroundReplacements(Code, FileAndReplacements.second,
                                              *Style);
        if (!Replacements) {
          llvm::errs() << llvm::toString(Replacements.takeError()) << "\n";
          continue;
        }
        if (llvm::Expected<tooling::Replacements> FormattedReplacements =
                format::formatReplacements(Code, *Replacements, *Style)) {
          Replacements = std::move(FormattedReplacements);
          if (!Replacements)
            llvm_unreachable("!Replacements");
        } else {
          llvm::errs() << llvm::toString(FormattedReplacements.takeError())
                       << ". Skipping formatting.\n";
        }
        if (!tooling::applyAllReplacements(Replacements.get(), Rewrite)) {
          llvm::errs() << "Can't apply replacements for file " << File << "\n";
        }
      }
      if (Rewrite.overwriteChangedFiles()) {
        llvm::errs() << "clang-tidy failed to apply suggested fixes.\n";
      } else {
        llvm::errs() << "clang-tidy applied " << AppliedFixes << " of "
                     << TotalFixes << " suggested fixes.\n";
      }
    }
  }

  unsigned getWarningsAsErrorsCount() const { return WarningsAsErrors; }

private:
  SourceLocation getLocation(StringRef FilePath, unsigned Offset) {
    if (FilePath.empty())
      return SourceLocation();

    const FileEntry *File = SourceMgr.getFileManager().getFile(FilePath);
    FileID ID = SourceMgr.getOrCreateFileID(File, SrcMgr::C_User);
    return SourceMgr.getLocForStartOfFile(ID).getLocWithOffset(Offset);
  }

  void reportNote(const tooling::DiagnosticMessage &Message) {
    SourceLocation Loc = getLocation(Message.FilePath, Message.FileOffset);
    Diags.Report(Loc, Diags.getCustomDiagID(DiagnosticsEngine::Note, "%0"))
        << Message.Message;
  }

  FileManager Files;
  LangOptions LangOpts; // FIXME: use langopts from each original file
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts;
  DiagnosticConsumer *DiagPrinter;
  DiagnosticsEngine Diags;
  SourceManager SourceMgr;
  llvm::StringMap<Replacements> FileReplacements;
  ClangTidyContext &Context;
  bool ApplyFixes;
  unsigned TotalFixes;
  unsigned AppliedFixes;
  unsigned WarningsAsErrors;
};

class ClangTidyASTConsumer : public MultiplexConsumer {
public:
  ClangTidyASTConsumer(std::vector<std::unique_ptr<ASTConsumer>> Consumers,
                       std::unique_ptr<ClangTidyProfiling> Profiling,
                       std::unique_ptr<ast_matchers::MatchFinder> Finder,
                       std::vector<std::unique_ptr<ClangTidyCheck>> Checks)
      : MultiplexConsumer(std::move(Consumers)),
        Profiling(std::move(Profiling)), Finder(std::move(Finder)),
        Checks(std::move(Checks)) {}

private:
  // Destructor order matters! Profiling must be destructed last.
  // Or at least after Finder.
  std::unique_ptr<ClangTidyProfiling> Profiling;
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

#if CLANG_ENABLE_STATIC_ANALYZER
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

typedef std::vector<std::pair<std::string, bool>> CheckersList;

static CheckersList getCheckersControlList(ClangTidyContext &Context,
                                           bool IncludeExperimental) {
  CheckersList List;

  const auto &RegisteredCheckers =
      AnalyzerOptions::getRegisteredCheckers(IncludeExperimental);
  bool AnalyzerChecksEnabled = false;
  for (StringRef CheckName : RegisteredCheckers) {
    std::string ClangTidyCheckName((AnalyzerCheckNamePrefix + CheckName).str());
    AnalyzerChecksEnabled |= Context.isCheckEnabled(ClangTidyCheckName);
  }

  if (!AnalyzerChecksEnabled)
    return List;

  // List all static analyzer checkers that our filter enables.
  //
  // Always add all core checkers if any other static analyzer check is enabled.
  // This is currently necessary, as other path sensitive checks rely on the
  // core checkers.
  for (StringRef CheckName : RegisteredCheckers) {
    std::string ClangTidyCheckName((AnalyzerCheckNamePrefix + CheckName).str());

    if (CheckName.startswith("core") ||
        Context.isCheckEnabled(ClangTidyCheckName)) {
      List.emplace_back(CheckName, true);
    }
  }
  return List;
}
#endif // CLANG_ENABLE_STATIC_ANALYZER

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

  std::unique_ptr<ClangTidyProfiling> Profiling;
  if (Context.getEnableProfiling()) {
    Profiling = llvm::make_unique<ClangTidyProfiling>(
        Context.getProfileStorageParams());
    FinderOptions.CheckProfiling.emplace(Profiling->Records);
  }

  std::unique_ptr<ast_matchers::MatchFinder> Finder(
      new ast_matchers::MatchFinder(std::move(FinderOptions)));

  for (auto &Check : Checks) {
    Check->registerMatchers(&*Finder);
    Check->registerPPCallbacks(Compiler);
  }

  std::vector<std::unique_ptr<ASTConsumer>> Consumers;
  if (!Checks.empty())
    Consumers.push_back(Finder->newASTConsumer());

#if CLANG_ENABLE_STATIC_ANALYZER
  AnalyzerOptionsRef AnalyzerOptions = Compiler.getAnalyzerOpts();
  AnalyzerOptions->CheckersControlList =
      getCheckersControlList(Context, Context.canEnableAnalyzerAlphaCheckers());
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
#endif // CLANG_ENABLE_STATIC_ANALYZER
  return llvm::make_unique<ClangTidyASTConsumer>(
      std::move(Consumers), std::move(Profiling), std::move(Finder),
      std::move(Checks));
}

std::vector<std::string> ClangTidyASTConsumerFactory::getCheckNames() {
  std::vector<std::string> CheckNames;
  for (const auto &CheckFactory : *CheckFactories) {
    if (Context.isCheckEnabled(CheckFactory.first))
      CheckNames.push_back(CheckFactory.first);
  }

#if CLANG_ENABLE_STATIC_ANALYZER
  for (const auto &AnalyzerCheck : getCheckersControlList(
           Context, Context.canEnableAnalyzerAlphaCheckers()))
    CheckNames.push_back(AnalyzerCheckNamePrefix + AnalyzerCheck.first);
#endif // CLANG_ENABLE_STATIC_ANALYZER

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

DiagnosticBuilder ClangTidyCheck::diag(SourceLocation Loc, StringRef Message,
                                       DiagnosticIDs::Level Level) {
  return Context->diag(CheckName, Loc, Message, Level);
}

void ClangTidyCheck::run(const ast_matchers::MatchFinder::MatchResult &Result) {
  // For historical reasons, checks don't implement the MatchFinder run()
  // callback directly. We keep the run()/check() distinction to avoid interface
  // churn, and to allow us to add cross-cutting logic in the future.
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

std::vector<std::string>
getCheckNames(const ClangTidyOptions &Options,
              bool AllowEnablingAnalyzerAlphaCheckers) {
  clang::tidy::ClangTidyContext Context(
      llvm::make_unique<DefaultOptionsProvider>(ClangTidyGlobalOptions(),
                                                Options),
      AllowEnablingAnalyzerAlphaCheckers);
  ClangTidyASTConsumerFactory Factory(Context);
  return Factory.getCheckNames();
}

ClangTidyOptions::OptionMap
getCheckOptions(const ClangTidyOptions &Options,
                bool AllowEnablingAnalyzerAlphaCheckers) {
  clang::tidy::ClangTidyContext Context(
      llvm::make_unique<DefaultOptionsProvider>(ClangTidyGlobalOptions(),
                                                Options),
      AllowEnablingAnalyzerAlphaCheckers);
  ClangTidyASTConsumerFactory Factory(Context);
  return Factory.getCheckOptions();
}

std::vector<ClangTidyError>
runClangTidy(clang::tidy::ClangTidyContext &Context,
             const CompilationDatabase &Compilations,
             ArrayRef<std::string> InputFiles,
             llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> BaseFS,
             bool EnableCheckProfile, llvm::StringRef StoreCheckProfile) {
  ClangTool Tool(Compilations, InputFiles,
                 std::make_shared<PCHContainerOperations>(), BaseFS);

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

  Tool.appendArgumentsAdjuster(PerFileExtraArgumentsInserter);
  Tool.appendArgumentsAdjuster(getStripPluginsAdjuster());
  Context.setEnableProfiling(EnableCheckProfile);
  Context.setProfileStoragePrefix(StoreCheckProfile);

  ClangTidyDiagnosticConsumer DiagConsumer(Context);
  DiagnosticsEngine DE(new DiagnosticIDs(), new DiagnosticOptions(),
                       &DiagConsumer, /*ShouldOwnClient=*/false);
  Context.setDiagnosticsEngine(&DE);
  Tool.setDiagnosticConsumer(&DiagConsumer);

  class ActionFactory : public FrontendActionFactory {
  public:
    ActionFactory(ClangTidyContext &Context) : ConsumerFactory(Context) {}
    FrontendAction *create() override { return new Action(&ConsumerFactory); }

    bool runInvocation(std::shared_ptr<CompilerInvocation> Invocation,
                       FileManager *Files,
                       std::shared_ptr<PCHContainerOperations> PCHContainerOps,
                       DiagnosticConsumer *DiagConsumer) override {
      // Explicitly set ProgramAction to RunAnalysis to make the preprocessor
      // define __clang_analyzer__ macro. The frontend analyzer action will not
      // be called here.
      Invocation->getFrontendOpts().ProgramAction = frontend::RunAnalysis;
      return FrontendActionFactory::runInvocation(
          Invocation, Files, PCHContainerOps, DiagConsumer);
    }

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
  return DiagConsumer.take();
}

void handleErrors(llvm::ArrayRef<ClangTidyError> Errors,
                  ClangTidyContext &Context, bool Fix,
                  unsigned &WarningsAsErrorsCount,
                  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> BaseFS) {
  ErrorReporter Reporter(Context, Fix, BaseFS);
  llvm::vfs::FileSystem &FileSystem =
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

void exportReplacements(const llvm::StringRef MainFilePath,
                        const std::vector<ClangTidyError> &Errors,
                        raw_ostream &OS) {
  TranslationUnitDiagnostics TUD;
  TUD.MainSourceFile = MainFilePath;
  for (const auto &Error : Errors) {
    tooling::Diagnostic Diag = Error;
    TUD.Diagnostics.insert(TUD.Diagnostics.end(), Diag);
  }

  yaml::Output YAML(OS);
  YAML << TUD;
}

} // namespace tidy
} // namespace clang
