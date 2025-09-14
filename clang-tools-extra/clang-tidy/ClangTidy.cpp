//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "ClangTidyCheck.h"
#include "ClangTidyDiagnosticConsumer.h"
#include "ClangTidyModuleRegistry.h"
#include "ClangTidyProfiling.h"
#include "ExpandModularHeadersPPCallbacks.h"
#include "clang-tidy-config.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Format/Format.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Rewrite/Frontend/FixItRewriter.h"
#include "clang/Tooling/Core/Diagnostic.h"
#include "clang/Tooling/DiagnosticsYaml.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/Process.h"
#include <utility>

#if CLANG_TIDY_ENABLE_STATIC_ANALYZER
#include "clang/Analysis/PathDiagnostic.h"
#include "clang/StaticAnalyzer/Frontend/AnalysisConsumer.h"
#endif // CLANG_TIDY_ENABLE_STATIC_ANALYZER

using namespace clang::ast_matchers;
using namespace clang::driver;
using namespace clang::tooling;
using namespace llvm;

LLVM_INSTANTIATE_REGISTRY(clang::tidy::ClangTidyModuleRegistry)

namespace clang::tidy {

namespace {
#if CLANG_TIDY_ENABLE_STATIC_ANALYZER
#define ANALYZER_CHECK_NAME_PREFIX "clang-analyzer-"
static constexpr llvm::StringLiteral AnalyzerCheckNamePrefix =
    ANALYZER_CHECK_NAME_PREFIX;

class AnalyzerDiagnosticConsumer : public ento::PathDiagnosticConsumer {
public:
  AnalyzerDiagnosticConsumer(ClangTidyContext &Context) : Context(Context) {}

  void FlushDiagnosticsImpl(std::vector<const ento::PathDiagnostic *> &Diags,
                            FilesMade *FilesMade) override {
    for (const ento::PathDiagnostic *PD : Diags) {
      SmallString<64> CheckName(AnalyzerCheckNamePrefix);
      CheckName += PD->getCheckerName();
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
#endif // CLANG_TIDY_ENABLE_STATIC_ANALYZER

class ErrorReporter {
public:
  ErrorReporter(ClangTidyContext &Context, FixBehaviour ApplyFixes,
                llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> BaseFS)
      : Files(FileSystemOptions(), std::move(BaseFS)),
        DiagPrinter(new TextDiagnosticPrinter(llvm::outs(), DiagOpts)),
        Diags(DiagnosticIDs::create(), DiagOpts, DiagPrinter),
        SourceMgr(Diags, Files), Context(Context), ApplyFixes(ApplyFixes) {
    DiagOpts.ShowColors = Context.getOptions().UseColor.value_or(
        llvm::sys::Process::StandardOutHasColors());
    DiagPrinter->BeginSourceFile(LangOpts);
    if (DiagOpts.ShowColors && !llvm::sys::Process::StandardOutIsDisplayed()) {
      llvm::sys::Process::UseANSIEscapeCodes(true);
    }
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
      if (!Error.EnabledDiagnosticAliases.empty())
        Name += "," + llvm::join(Error.EnabledDiagnosticAliases, ",");
      if (Error.IsWarningAsError) {
        Name += ",-warnings-as-errors";
        Level = DiagnosticsEngine::Error;
        WarningsAsErrors++;
      }
      auto Diag = Diags.Report(Loc, Diags.getCustomDiagID(Level, "%0 [%1]"))
                  << Message.Message << Name;
      for (const FileByteRange &FBR : Error.Message.Ranges)
        Diag << getRange(FBR);
      // FIXME: explore options to support interactive fix selection.
      const llvm::StringMap<Replacements> *ChosenFix = nullptr;
      if (ApplyFixes != FB_NoFix &&
          (ChosenFix = getFixIt(Error, ApplyFixes == FB_FixNotes))) {
        for (const auto &FileAndReplacements : *ChosenFix) {
          for (const auto &Repl : FileAndReplacements.second) {
            ++TotalFixes;
            bool CanBeApplied = false;
            if (!Repl.isApplicable())
              continue;
            SourceLocation FixLoc;
            SmallString<128> FixAbsoluteFilePath = Repl.getFilePath();
            Files.makeAbsolutePath(FixAbsoluteFilePath);
            tooling::Replacement R(FixAbsoluteFilePath, Repl.getOffset(),
                                   Repl.getLength(), Repl.getReplacementText());
            auto &Entry = FileReplacements[R.getFilePath()];
            Replacements &Replacements = Entry.Replaces;
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
            FixLoc = getLocation(FixAbsoluteFilePath, Repl.getOffset());
            FixLocations.push_back(std::make_pair(FixLoc, CanBeApplied));
            Entry.BuildDir = Error.BuildDirectory;
          }
        }
      }
      reportFix(Diag, Error.Message.Fix);
    }
    for (auto Fix : FixLocations) {
      Diags.Report(Fix.first, Fix.second ? diag::note_fixit_applied
                                         : diag::note_fixit_failed);
    }
    for (const auto &Note : Error.Notes)
      reportNote(Note);
  }

  void finish() {
    if (TotalFixes > 0) {
      auto &VFS = Files.getVirtualFileSystem();
      auto OriginalCWD = VFS.getCurrentWorkingDirectory();
      bool AnyNotWritten = false;

      for (const auto &FileAndReplacements : FileReplacements) {
        Rewriter Rewrite(SourceMgr, LangOpts);
        StringRef File = FileAndReplacements.first();
        VFS.setCurrentWorkingDirectory(FileAndReplacements.second.BuildDir);
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
            format::cleanupAroundReplacements(
                Code, FileAndReplacements.second.Replaces, *Style);
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
        AnyNotWritten |= Rewrite.overwriteChangedFiles();
      }

      if (AnyNotWritten) {
        llvm::errs() << "clang-tidy failed to apply suggested fixes.\n";
      } else {
        llvm::errs() << "clang-tidy applied " << AppliedFixes << " of "
                     << TotalFixes << " suggested fixes.\n";
      }

      if (OriginalCWD)
        VFS.setCurrentWorkingDirectory(*OriginalCWD);
    }
  }

  unsigned getWarningsAsErrorsCount() const { return WarningsAsErrors; }

private:
  SourceLocation getLocation(StringRef FilePath, unsigned Offset) {
    if (FilePath.empty())
      return {};

    auto File = SourceMgr.getFileManager().getOptionalFileRef(FilePath);
    if (!File)
      return {};

    FileID ID = SourceMgr.getOrCreateFileID(*File, SrcMgr::C_User);
    return SourceMgr.getLocForStartOfFile(ID).getLocWithOffset(Offset);
  }

  void reportFix(const DiagnosticBuilder &Diag,
                 const llvm::StringMap<Replacements> &Fix) {
    for (const auto &FileAndReplacements : Fix) {
      for (const auto &Repl : FileAndReplacements.second) {
        if (!Repl.isApplicable())
          continue;
        FileByteRange FBR;
        FBR.FilePath = Repl.getFilePath().str();
        FBR.FileOffset = Repl.getOffset();
        FBR.Length = Repl.getLength();

        Diag << FixItHint::CreateReplacement(getRange(FBR),
                                             Repl.getReplacementText());
      }
    }
  }

  void reportNote(const tooling::DiagnosticMessage &Message) {
    SourceLocation Loc = getLocation(Message.FilePath, Message.FileOffset);
    auto Diag =
        Diags.Report(Loc, Diags.getCustomDiagID(DiagnosticsEngine::Note, "%0"))
        << Message.Message;
    for (const FileByteRange &FBR : Message.Ranges)
      Diag << getRange(FBR);
    reportFix(Diag, Message.Fix);
  }

  CharSourceRange getRange(const FileByteRange &Range) {
    SmallString<128> AbsoluteFilePath{Range.FilePath};
    Files.makeAbsolutePath(AbsoluteFilePath);
    SourceLocation BeginLoc = getLocation(AbsoluteFilePath, Range.FileOffset);
    SourceLocation EndLoc = BeginLoc.getLocWithOffset(Range.Length);
    // Retrieve the source range for applicable highlights and fixes. Macro
    // definition on the command line have locations in a virtual buffer and
    // don't have valid file paths and are therefore not applicable.
    return CharSourceRange::getCharRange(BeginLoc, EndLoc);
  }

  struct ReplacementsWithBuildDir {
    StringRef BuildDir;
    Replacements Replaces;
  };

  FileManager Files;
  LangOptions LangOpts; // FIXME: use langopts from each original file
  DiagnosticOptions DiagOpts;
  DiagnosticConsumer *DiagPrinter;
  DiagnosticsEngine Diags;
  SourceManager SourceMgr;
  llvm::StringMap<ReplacementsWithBuildDir> FileReplacements;
  ClangTidyContext &Context;
  FixBehaviour ApplyFixes;
  unsigned TotalFixes = 0U;
  unsigned AppliedFixes = 0U;
  unsigned WarningsAsErrors = 0U;
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
  void anchor() override {};
};

} // namespace

ClangTidyASTConsumerFactory::ClangTidyASTConsumerFactory(
    ClangTidyContext &Context,
    IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> OverlayFS)
    : Context(Context), OverlayFS(std::move(OverlayFS)),
      CheckFactories(new ClangTidyCheckFactories) {
  for (ClangTidyModuleRegistry::entry E : ClangTidyModuleRegistry::entries()) {
    std::unique_ptr<ClangTidyModule> Module = E.instantiate();
    Module->addCheckFactories(*CheckFactories);
  }
}

#if CLANG_TIDY_ENABLE_STATIC_ANALYZER
static void
setStaticAnalyzerCheckerOpts(const ClangTidyOptions &Opts,
                             clang::AnalyzerOptions &AnalyzerOptions) {
  for (const auto &Opt : Opts.CheckOptions) {
    StringRef OptName(Opt.getKey());
    if (!OptName.consume_front(AnalyzerCheckNamePrefix))
      continue;
    // Analyzer options are always local options so we can ignore priority.
    AnalyzerOptions.Config[OptName] = Opt.getValue().Value;
  }
}

using CheckersList = std::vector<std::pair<std::string, bool>>;

static CheckersList getAnalyzerCheckersAndPackages(ClangTidyContext &Context,
                                                   bool IncludeExperimental) {
  CheckersList List;

  const auto &RegisteredCheckers =
      AnalyzerOptions::getRegisteredCheckers(IncludeExperimental);
  const bool AnalyzerChecksEnabled =
      llvm::any_of(RegisteredCheckers, [&](StringRef CheckName) -> bool {
        return Context.isCheckEnabled(
            (AnalyzerCheckNamePrefix + CheckName).str());
      });

  if (!AnalyzerChecksEnabled)
    return List;

  // List all static analyzer checkers that our filter enables.
  //
  // Always add all core checkers if any other static analyzer check is enabled.
  // This is currently necessary, as other path sensitive checks rely on the
  // core checkers.
  for (StringRef CheckName : RegisteredCheckers) {
    std::string ClangTidyCheckName((AnalyzerCheckNamePrefix + CheckName).str());

    if (CheckName.starts_with("core") ||
        Context.isCheckEnabled(ClangTidyCheckName)) {
      List.emplace_back(std::string(CheckName), true);
    }
  }
  return List;
}
#endif // CLANG_TIDY_ENABLE_STATIC_ANALYZER

std::unique_ptr<clang::ASTConsumer>
ClangTidyASTConsumerFactory::createASTConsumer(
    clang::CompilerInstance &Compiler, StringRef File) {
  // FIXME: Move this to a separate method, so that CreateASTConsumer doesn't
  // modify Compiler.
  SourceManager *SM = &Compiler.getSourceManager();
  Context.setSourceManager(SM);
  Context.setCurrentFile(File);
  Context.setASTContext(&Compiler.getASTContext());

  auto WorkingDir = Compiler.getSourceManager()
                        .getFileManager()
                        .getVirtualFileSystem()
                        .getCurrentWorkingDirectory();
  if (WorkingDir)
    Context.setCurrentBuildDirectory(WorkingDir.get());

  std::vector<std::unique_ptr<ClangTidyCheck>> Checks =
      CheckFactories->createChecksForLanguage(&Context);

  ast_matchers::MatchFinder::MatchFinderOptions FinderOptions;

  std::unique_ptr<ClangTidyProfiling> Profiling;
  if (Context.getEnableProfiling()) {
    Profiling =
        std::make_unique<ClangTidyProfiling>(Context.getProfileStorageParams());
    FinderOptions.CheckProfiling.emplace(Profiling->Records);
  }

  // Avoid processing system headers, unless the user explicitly requests it
  if (!Context.getOptions().SystemHeaders.value_or(false))
    FinderOptions.IgnoreSystemHeaders = true;

  std::unique_ptr<ast_matchers::MatchFinder> Finder(
      new ast_matchers::MatchFinder(std::move(FinderOptions)));

  Preprocessor *PP = &Compiler.getPreprocessor();
  Preprocessor *ModuleExpanderPP = PP;

  if (Context.canEnableModuleHeadersParsing() &&
      Context.getLangOpts().Modules && OverlayFS != nullptr) {
    auto ModuleExpander =
        std::make_unique<ExpandModularHeadersPPCallbacks>(&Compiler, OverlayFS);
    ModuleExpanderPP = ModuleExpander->getPreprocessor();
    PP->addPPCallbacks(std::move(ModuleExpander));
  }

  for (auto &Check : Checks) {
    Check->registerMatchers(&*Finder);
    Check->registerPPCallbacks(*SM, PP, ModuleExpanderPP);
  }

  std::vector<std::unique_ptr<ASTConsumer>> Consumers;
  if (!Checks.empty())
    Consumers.push_back(Finder->newASTConsumer());

#if CLANG_TIDY_ENABLE_STATIC_ANALYZER
  AnalyzerOptions &AnalyzerOptions = Compiler.getAnalyzerOpts();
  AnalyzerOptions.CheckersAndPackages = getAnalyzerCheckersAndPackages(
      Context, Context.canEnableAnalyzerAlphaCheckers());
  if (!AnalyzerOptions.CheckersAndPackages.empty()) {
    setStaticAnalyzerCheckerOpts(Context.getOptions(), AnalyzerOptions);
    AnalyzerOptions.AnalysisDiagOpt = PD_NONE;
    std::unique_ptr<ento::AnalysisASTConsumer> AnalysisConsumer =
        ento::CreateAnalysisConsumer(Compiler);
    AnalysisConsumer->AddDiagnosticConsumer(
        std::make_unique<AnalyzerDiagnosticConsumer>(Context));
    Consumers.push_back(std::move(AnalysisConsumer));
  }
#endif // CLANG_TIDY_ENABLE_STATIC_ANALYZER
  return std::make_unique<ClangTidyASTConsumer>(
      std::move(Consumers), std::move(Profiling), std::move(Finder),
      std::move(Checks));
}

std::vector<std::string> ClangTidyASTConsumerFactory::getCheckNames() {
  std::vector<std::string> CheckNames;
  for (const auto &CheckFactory : *CheckFactories) {
    if (Context.isCheckEnabled(CheckFactory.getKey()))
      CheckNames.emplace_back(CheckFactory.getKey());
  }

#if CLANG_TIDY_ENABLE_STATIC_ANALYZER
  for (const auto &AnalyzerCheck : getAnalyzerCheckersAndPackages(
           Context, Context.canEnableAnalyzerAlphaCheckers()))
    CheckNames.emplace_back(
        (AnalyzerCheckNamePrefix + AnalyzerCheck.first).str());
#endif // CLANG_TIDY_ENABLE_STATIC_ANALYZER

  llvm::sort(CheckNames);
  return CheckNames;
}

ClangTidyOptions::OptionMap ClangTidyASTConsumerFactory::getCheckOptions() {
  ClangTidyOptions::OptionMap Options;
  std::vector<std::unique_ptr<ClangTidyCheck>> Checks =
      CheckFactories->createChecks(&Context);
  for (const auto &Check : Checks)
    Check->storeOptions(Options);
  return Options;
}

std::vector<std::string>
getCheckNames(const ClangTidyOptions &Options,
              bool AllowEnablingAnalyzerAlphaCheckers) {
  clang::tidy::ClangTidyContext Context(
      std::make_unique<DefaultOptionsProvider>(ClangTidyGlobalOptions(),
                                               Options),
      AllowEnablingAnalyzerAlphaCheckers);
  ClangTidyASTConsumerFactory Factory(Context);
  return Factory.getCheckNames();
}

void filterCheckOptions(ClangTidyOptions &Options,
                        const std::vector<std::string> &EnabledChecks) {
  ClangTidyOptions::OptionMap FilteredOptions;
  for (const auto &[OptionName, Value] : Options.CheckOptions) {
    const size_t CheckNameEndPos = OptionName.find('.');
    if (CheckNameEndPos == StringRef::npos)
      continue;
    const StringRef CheckName = OptionName.substr(0, CheckNameEndPos);
    if (llvm::binary_search(EnabledChecks, CheckName))
      FilteredOptions[OptionName] = Value;
  }
  Options.CheckOptions = std::move(FilteredOptions);
}

ClangTidyOptions::OptionMap
getCheckOptions(const ClangTidyOptions &Options,
                bool AllowEnablingAnalyzerAlphaCheckers) {
  clang::tidy::ClangTidyContext Context(
      std::make_unique<DefaultOptionsProvider>(ClangTidyGlobalOptions(),
                                               Options),
      AllowEnablingAnalyzerAlphaCheckers);
  ClangTidyDiagnosticConsumer DiagConsumer(Context);
  auto DiagOpts = std::make_unique<DiagnosticOptions>();
  DiagnosticsEngine DE(llvm::makeIntrusiveRefCnt<DiagnosticIDs>(), *DiagOpts,
                       &DiagConsumer, /*ShouldOwnClient=*/false);
  Context.setDiagnosticsEngine(std::move(DiagOpts), &DE);
  ClangTidyASTConsumerFactory Factory(Context);
  return Factory.getCheckOptions();
}

std::vector<ClangTidyError>
runClangTidy(clang::tidy::ClangTidyContext &Context,
             const CompilationDatabase &Compilations,
             ArrayRef<std::string> InputFiles,
             llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> BaseFS,
             bool ApplyAnyFix, bool EnableCheckProfile,
             llvm::StringRef StoreCheckProfile, bool Quiet) {
  ClangTool Tool(Compilations, InputFiles,
                 std::make_shared<PCHContainerOperations>(), BaseFS);

  // Add extra arguments passed by the clang-tidy command-line.
  ArgumentsAdjuster PerFileExtraArgumentsInserter =
      [&Context](const CommandLineArguments &Args, StringRef Filename) {
        ClangTidyOptions Opts = Context.getOptionsForFile(Filename);
        CommandLineArguments AdjustedArgs = Args;
        if (Opts.ExtraArgsBefore) {
          auto I = AdjustedArgs.begin();
          if (I != AdjustedArgs.end() && !StringRef(*I).starts_with("-"))
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

  ClangTidyDiagnosticConsumer DiagConsumer(Context, nullptr, true, ApplyAnyFix);
  auto DiagOpts = std::make_unique<DiagnosticOptions>();
  DiagnosticsEngine DE(DiagnosticIDs::create(), *DiagOpts, &DiagConsumer,
                       /*ShouldOwnClient=*/false);
  Context.setDiagnosticsEngine(std::move(DiagOpts), &DE);
  Tool.setDiagnosticConsumer(&DiagConsumer);

  class ActionFactory : public FrontendActionFactory {
  public:
    ActionFactory(ClangTidyContext &Context,
                  IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> BaseFS,
                  bool Quiet)
        : ConsumerFactory(Context, std::move(BaseFS)), Quiet(Quiet) {}
    std::unique_ptr<FrontendAction> create() override {
      return std::make_unique<Action>(&ConsumerFactory);
    }

    bool runInvocation(std::shared_ptr<CompilerInvocation> Invocation,
                       FileManager *Files,
                       std::shared_ptr<PCHContainerOperations> PCHContainerOps,
                       DiagnosticConsumer *DiagConsumer) override {
      // Explicitly ask to define __clang_analyzer__ macro.
      Invocation->getPreprocessorOpts().SetUpStaticAnalyzer = true;
      if (Quiet)
        Invocation->getDiagnosticOpts().ShowCarets = false;
      return FrontendActionFactory::runInvocation(
          Invocation, Files, PCHContainerOps, DiagConsumer);
    }

  private:
    class Action : public ASTFrontendAction {
    public:
      Action(ClangTidyASTConsumerFactory *Factory) : Factory(Factory) {}
      std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &Compiler,
                                                     StringRef File) override {
        return Factory->createASTConsumer(Compiler, File);
      }

    private:
      ClangTidyASTConsumerFactory *Factory;
    };

    ClangTidyASTConsumerFactory ConsumerFactory;
    bool Quiet;
  };

  ActionFactory Factory(Context, std::move(BaseFS), Quiet);
  Tool.run(&Factory);
  return DiagConsumer.take();
}

void handleErrors(llvm::ArrayRef<ClangTidyError> Errors,
                  ClangTidyContext &Context, FixBehaviour Fix,
                  unsigned &WarningsAsErrorsCount,
                  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> BaseFS) {
  ErrorReporter Reporter(Context, Fix, std::move(BaseFS));
  llvm::vfs::FileSystem &FileSystem =
      Reporter.getSourceManager().getFileManager().getVirtualFileSystem();
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
  Reporter.finish();
  WarningsAsErrorsCount += Reporter.getWarningsAsErrorsCount();
}

void exportReplacements(const llvm::StringRef MainFilePath,
                        const std::vector<ClangTidyError> &Errors,
                        raw_ostream &OS) {
  TranslationUnitDiagnostics TUD;
  TUD.MainSourceFile = std::string(MainFilePath);
  for (const auto &Error : Errors) {
    tooling::Diagnostic Diag = Error;
    if (Error.IsWarningAsError)
      Diag.DiagLevel = tooling::Diagnostic::Error;
    TUD.Diagnostics.insert(TUD.Diagnostics.end(), Diag);
  }

  yaml::Output YAML(OS);
  YAML << TUD;
}

ChecksAndOptions
getAllChecksAndOptions(bool AllowEnablingAnalyzerAlphaCheckers) {
  ChecksAndOptions Result;
  ClangTidyOptions Opts;
  Opts.Checks = "*";
  clang::tidy::ClangTidyContext Context(
      std::make_unique<DefaultOptionsProvider>(ClangTidyGlobalOptions(), Opts),
      AllowEnablingAnalyzerAlphaCheckers);
  ClangTidyCheckFactories Factories;
  for (const ClangTidyModuleRegistry::entry &Module :
       ClangTidyModuleRegistry::entries()) {
    Module.instantiate()->addCheckFactories(Factories);
  }

  for (const auto &Factory : Factories)
    Result.Checks.insert(Factory.getKey());

#if CLANG_TIDY_ENABLE_STATIC_ANALYZER
  SmallString<64> Buffer(AnalyzerCheckNamePrefix);
  size_t DefSize = Buffer.size();
  for (const auto &AnalyzerCheck : AnalyzerOptions::getRegisteredCheckers(
           AllowEnablingAnalyzerAlphaCheckers)) {
    Buffer.truncate(DefSize);
    Buffer.append(AnalyzerCheck);
    Result.Checks.insert(Buffer);
  }

  static constexpr llvm::StringLiteral OptionNames[] = {
#define GET_CHECKER_OPTIONS
#define CHECKER_OPTION(TYPE, CHECKER, OPTION_NAME, DESCRIPTION, DEFAULT,       \
                       RELEASE, HIDDEN)                                        \
  ANALYZER_CHECK_NAME_PREFIX CHECKER ":" OPTION_NAME,

#include "clang/StaticAnalyzer/Checkers/Checkers.inc"
#undef CHECKER_OPTION
#undef GET_CHECKER_OPTIONS
  };

  Result.Options.insert_range(OptionNames);
#endif // CLANG_TIDY_ENABLE_STATIC_ANALYZER

  Context.setOptionsCollector(&Result.Options);
  for (const auto &Factory : Factories) {
    Factory.getValue()(Factory.getKey(), &Context);
  }

  return Result;
}
} // namespace clang::tidy
