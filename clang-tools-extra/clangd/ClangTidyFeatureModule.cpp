//===--- ClangTidyFeatureModule.cpp - clang-tidy integration -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangTidyFeatureModule.h"
#include "../clang-tidy/ClangTidyCheck.h"
#include "../clang-tidy/ClangTidyDiagnosticConsumer.h"
#include "../clang-tidy/ClangTidyModule.h"
#include "../clang-tidy/ClangTidyOptions.h"
#include "Config.h"
#include "Diagnostics.h"
#include "Feature.h"
#include "SourceCode.h"
#include "support/Logger.h"
#include "support/Trace.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/FileManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Tooling/Core/Diagnostic.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// Force the linker to link in Clang-tidy modules.
// clangd doesn't support the static analyzer.
#if CLANGD_TIDY_CHECKS
#define CLANG_TIDY_DISABLE_STATIC_ANALYZER_CHECKS
#include "../clang-tidy/ClangTidyForceLinker.h"
#endif

#if CLANG_TIDY_ENABLE_QUERY_BASED_CUSTOM_CHECKS
#include "../clang-tidy/ClangTidy.h"
#include <mutex>
#endif

namespace clang {
namespace clangd {
namespace {

// Filter for clang diagnostics groups enabled by CTOptions.Checks.
//
// These are check names like clang-diagnostics-unused.
// Note that unlike -Wunused, clang-diagnostics-unused does not imply
// subcategories like clang-diagnostics-unused-function.
//
// This is used to determine which diagnostics can be enabled by ExtraArgs in
// the clang-tidy configuration.
class TidyDiagnosticGroups {
  // Whether all diagnostic groups are enabled by default.
  // True if we've seen clang-diagnostic-*.
  bool Default = false;
  // Set of diag::Group whose enablement != Default.
  // If Default is false, this is foo where we've seen clang-diagnostic-foo.
  llvm::DenseSet<unsigned> Exceptions;

public:
  TidyDiagnosticGroups(llvm::StringRef Checks) {
    constexpr llvm::StringLiteral Prefix = "clang-diagnostic-";
    llvm::StringRef Check;
    while (!Checks.empty()) {
      std::tie(Check, Checks) = Checks.split(',');
      Check = Check.trim();
      if (Check.empty())
        continue;

      bool Enable = !Check.consume_front("-");
      bool Glob = Check.consume_back("*");
      if (Glob) {
        // Is this clang-diagnostic-*, or *, or so?
        // (We ignore all other types of globs).
        if (Prefix.starts_with(Check)) {
          Default = Enable;
          Exceptions.clear();
        }
        continue;
      }
      // In "*,clang-diagnostic-foo", the latter is a no-op.
      // The only non-glob entries we care about are clang-diagnostic-foo.
      if (Default == Enable || !Check.consume_front(Prefix))
        continue;
      if (auto Group = DiagnosticIDs::getGroupForWarningOption(Check))
        Exceptions.insert(static_cast<unsigned>(*Group));
    }
  }

  bool operator()(diag::Group Group) const {
    return Exceptions.contains(static_cast<unsigned>(Group)) ? !Default
                                                             : Default;
  }
};

// Find -W<group> and -Wno-<group> options in ExtraArgs and apply them to Diags.
//
// This is used to handle ExtraArgs in clang-tidy configuration.
// We don't use clang's standard handling of this as we want slightly different
// behavior (e.g. we want to exclude these from -Wno-error).
void applyWarningOptions(llvm::ArrayRef<std::string> ExtraArgs,
                         llvm::function_ref<bool(diag::Group)> EnabledGroups,
                         DiagnosticsEngine &Diags) {
  for (llvm::StringRef Group : ExtraArgs) {
    // Only handle args that are of the form -W[no-]<group>.
    // Other flags are possible but rare and deliberately out of scope.
    llvm::SmallVector<diag::kind> Members;
    if (!Group.consume_front("-W") || Group.empty())
      continue;
    bool Enable = !Group.consume_front("no-");
    if (Diags.getDiagnosticIDs()->getDiagnosticsInGroup(
            diag::Flavor::WarningOrError, Group, Members))
      continue;

    // Upgrade (or downgrade) the severity of each diagnostic in the group.
    // If -Werror is on, newly added warnings will be treated as errors.
    // We don't want this, so keep track of them to fix afterwards.
    bool NeedsWerrorExclusion = false;
    for (diag::kind ID : Members) {
      if (Enable) {
        if (Diags.getDiagnosticLevel(ID, SourceLocation()) >=
            DiagnosticsEngine::Warning)
          continue;
        auto DiagGroup = Diags.getDiagnosticIDs()->getGroupForDiag(ID);
        if (!DiagGroup || !EnabledGroups(*DiagGroup))
          continue;
        Diags.setSeverity(ID, diag::Severity::Warning, SourceLocation());
        NeedsWerrorExclusion |= Diags.getWarningsAsErrors();
      } else {
        Diags.setSeverity(ID, diag::Severity::Ignored, SourceLocation());
      }
    }
    if (NeedsWerrorExclusion) {
      // FIXME: there's no API to suppress -Werror for single diagnostics.
      // In some cases with sub-groups, we may end up erroneously
      // downgrading diagnostics that were -Werror in the compile command.
      Diags.setDiagnosticGroupWarningAsError(Group, false);
    }
  }
}

tidy::ClangTidyCheckFactories
filterFastChecks(const tidy::ClangTidyCheckFactories &All,
                 Config::FastCheckPolicy Policy) {
  if (Policy == Config::FastCheckPolicy::None)
    return All;
  bool AllowUnknown = Policy == Config::FastCheckPolicy::Loose;
  tidy::ClangTidyCheckFactories Fast;
  for (const auto &Factory : All)
    if (isFastTidyCheck(Factory.getKey()).value_or(AllowUnknown))
      Fast.registerCheckFactory(Factory.first(), Factory.second);
  return Fast;
}

class TidyASTListener final : public FeatureModule::ASTListener {
public:
  explicit TidyASTListener(std::shared_ptr<const TidyProvider> OptionsProvider)
      : Provider(std::move(OptionsProvider)) {}

  void beforeBeginSourceFile(CompilerInstance &CI) override {
    // Warning options must precede frontend initialization: module loading in
    // BeginSourceFile() can consult their severity to decide whether to fail.
    const auto &Inputs = CI.getFrontendOpts().Inputs;
    if (Inputs.empty() || !Inputs.front().isFile()) {
      elog("Cannot configure clang-tidy without a main-file input");
      return;
    }
    llvm::SmallString<256> Path(Inputs.front().getFile());
    // The remapped buffer carries clangd's original main-file spelling.
    for (const auto &Remap :
         llvm::reverse(CI.getPreprocessorOpts().RemappedFileBuffers)) {
      if (Remap.first == Inputs.front().getFile()) {
        Path = Remap.second->getBufferIdentifier();
        break;
      }
    }
    // Providers search parent directories for .clang-tidy and require an
    // absolute path. Respect the compiler's working directory, not clangd's.
    if (!llvm::sys::path::is_absolute(Path))
      CI.getFileManager().makeAbsolutePath(Path, /*Canonicalize=*/true);
    if (!llvm::sys::path::is_absolute(Path)) {
      elog("Cannot resolve clang-tidy input path: {0}", Path);
      return;
    }
    Filename = Path.str().str();
    trace::Span Tracer("ClangTidyOpts");
    Options = getTidyOptionsForFile(*Provider, Filename);
    dlog("ClangTidy configuration for file {0}: {1}", Filename,
         tidy::configurationAsText(*Options));

    // If clang-tidy is configured to emit clang warnings, we should too.
    //
    // Such clang-tidy configuration consists of two parts:
    //   - ExtraArgs: ["-Wfoo"] causes clang to produce the warnings
    //   - Checks: "clang-diagnostic-foo" prevents clang-tidy filtering them out
    //
    // In clang-tidy, diagnostics are emitted if they pass both checks.
    // When groups contain subgroups, -Wparent includes the child, but
    // clang-diagnostic-parent does not.
    //
    // We *don't* want to change the compile command directly. This can have
    // too many unexpected effects: breaking the command, interactions with
    // -- and -Werror, etc. Besides, we've already parsed the command.
    // Instead we parse the -W<group> flags and handle them directly.
    //
    // Similarly, we don't want to use Checks to filter clang diagnostics after
    // they are generated, as this spreads clang-tidy emulation everywhere.
    // Instead, we just use these to filter which extra diagnostics we enable.
    TidyDiagnosticGroups EnabledGroups(Options->Checks ? *Options->Checks
                                                       : llvm::StringRef());
    if (Options->ExtraArgsBefore)
      applyWarningOptions(*Options->ExtraArgsBefore, EnabledGroups,
                          CI.getDiagnostics());
    if (Options->ExtraArgs)
      applyWarningOptions(*Options->ExtraArgs, EnabledGroups,
                          CI.getDiagnostics());
  }

  void beforePPCallbacks(CompilerInstance &CI) override {
    // Preamble builds don't call beforeBeginSourceFile(). Running tidy there
    // would duplicate diagnostics and interfere with PCH generation.
    if (!Options)
      return;

    // Set up ClangTidy. Must happen after BeginSourceFile() so ASTContext
    // exists. Clang-tidy has some limitations to ensure reasonable performance:
    //  - checks don't see all preprocessor events in the preamble
    //  - matchers run only over the main-file top-level decls (and can't see
    //    ancestors outside this scope).
    // In practice almost all checks work well without modifications.
    trace::Span Tracer("ClangTidyInit");
    static const auto *AllFactories = [] {
      auto *Factories = new tidy::ClangTidyCheckFactories;
      for (const auto &Entry : tidy::ClangTidyModuleRegistry::entries())
        Entry.instantiate()->addCheckFactories(*Factories);
      return Factories;
    }();
    Context.emplace(
        std::make_unique<tidy::DefaultOptionsProvider>(
            tidy::ClangTidyGlobalOptions(), std::move(*Options)),
        /*AllowEnablingAnalyzerAlphaCheckers=*/false,
        /*EnableModuleHeadersParsing=*/false,
        Config::current().Diagnostics.ClangTidy.ExperimentalCustomChecks);
    Options.reset();
    // The lifetime of DiagnosticOptions is managed by \c Clang.
    Context->setDiagnosticsEngine(nullptr, &CI.getDiagnostics());
    Context->setASTContext(&CI.getASTContext());
    Context->setCurrentFile(Filename);
    Context->setSelfContainedDiags(true);
    tidy::ClangTidyCheckFactories Factories = *AllFactories;
#if CLANG_TIDY_ENABLE_QUERY_BASED_CUSTOM_CHECKS
    if (Context->canExperimentalCustomChecks() &&
        tidy::custom::RegisterCustomChecks) {
      // RegisterCustomChecks tracks names in process-wide mutable state.
      // Serializing its use keeps concurrent AST builds independent.
      static std::mutex CustomChecksMu;
      std::lock_guard<std::mutex> Lock(CustomChecksMu);
      tidy::custom::RegisterCustomChecks(Context->getOptions(), Factories);
    }
#endif
    auto FastFactories = filterFastChecks(
        Factories, Config::current().Diagnostics.ClangTidy.FastCheckFilter);
    Checks = FastFactories.createChecksForLanguage(&*Context);

    Preprocessor *PP = &CI.getPreprocessor();
    for (const auto &Check : Checks) {
      Check->registerPPCallbacks(CI.getSourceManager(), PP, PP);
      Check->registerMatchers(&Finder);
    }
  }

  void afterExecute(CompilerInstance &CI) override {
    if (Checks.empty())
      return;
    // Run the AST-dependent part of the clang-tidy checks.
    // (The preprocessor part ran already, via PPCallbacks).
    trace::Span Tracer("ClangTidyMatch");
    Finder.matchAST(CI.getASTContext());
  }

  void sawDiagnostic(const clang::Diagnostic &Info, clangd::Diag &D) override {
    // Preserve the previous suppression policy when no checks are active.
    if (Checks.empty())
      return;
    std::string CheckName = Context->getCheckName(Info.getID());
    if (CheckName.empty())
      return;

    if (Config::current().Diagnostics.Suppress.contains(CheckName)) {
      D.Severity = DiagnosticsEngine::Ignored;
      return;
    }

    // Check for suppression comment. Skip the check for diagnostics not
    // in the main file, because we don't want that function to query the
    // source buffer for preamble files. For the same reason, we ask
    // shouldSuppressDiagnostic to avoid I/O.
    // We let suppression comments take precedence over warning-as-error
    // to match clang-tidy's behaviour.
    bool InsideMainFile =
        Info.hasSourceManager() &&
        isInsideMainFile(Info.getLocation(), Info.getSourceManager());
    llvm::SmallVector<tooling::Diagnostic, 1> SuppressionErrors;
    if (InsideMainFile &&
        Context->shouldSuppressDiagnostic(D.Severity, Info, SuppressionErrors,
                                          /*AllowIO=*/false,
                                          /*EnableNolintBlocks=*/true)) {
      // FIXME: should we expose the suppression error (invalid use of
      // NOLINT comments)?
      D.Severity = DiagnosticsEngine::Ignored;
      return;
    }
    if (!Context->getOptions().SystemHeaders.value_or(false) &&
        Info.hasSourceManager() &&
        Info.getSourceManager().isInSystemMacro(Info.getLocation())) {
      D.Severity = DiagnosticsEngine::Ignored;
      return;
    }
    // Check for warning-as-error.
    if (D.Severity == DiagnosticsEngine::Warning &&
        Context->treatAsError(CheckName))
      D.Severity = DiagnosticsEngine::Error;

    // Compiler warnings keep their clang source and -W name, even when their
    // severity or suppression is controlled by clang-tidy options.
    if (!Context->isCompilerDiagnostic(Info.getID())) {
      D.Name = std::move(CheckName);
      D.Source = Diag::ClangTidy;
    }
  }

  void finalizeDiagnostic(clangd::Diag &D) override {
    // Constructor diagnostics precede check initialization.
    if (D.Source == Diag::Unknown && Context &&
        !Context->isCompilerDiagnostic(D.ID)) {
      auto Name = Context->getCheckName(D.ID);
      if (!Name.empty()) {
        D.Name = std::move(Name);
        D.Source = Diag::ClangTidy;
      }
    }
    if (D.Source != Diag::ClangTidy)
      return;
    // clang-tidy bakes the name into diagnostic messages. Strip it out.
    // It would be much nicer to make clang-tidy not do this.
    auto CleanMessage = [&](std::string &Message) {
      llvm::StringRef Rest(Message);
      if (Rest.consume_back("]") && Rest.consume_back(D.Name) &&
          Rest.consume_back(" ["))
        Message.resize(Rest.size());
    };
    CleanMessage(D.Message);
    for (auto &Note : D.Notes)
      CleanMessage(Note.Message);
    for (auto &Fix : D.Fixes)
      CleanMessage(Fix.Message);
    if (llvm::StringRef(D.Name).starts_with("misc-unused-") &&
        !llvm::is_contained(D.Tags, DiagnosticTag::Unnecessary))
      D.Tags.push_back(DiagnosticTag::Unnecessary);
    if (llvm::StringRef(D.Name).starts_with("modernize-") &&
        !llvm::is_contained(D.Tags, DiagnosticTag::Deprecated))
      D.Tags.push_back(DiagnosticTag::Deprecated);
  }

private:
  std::shared_ptr<const TidyProvider> Provider;
  std::string Filename;
  std::optional<tidy::ClangTidyOptions> Options;
  // Destruction order matters: Finder and checks refer to Context.
  std::optional<tidy::ClangTidyContext> Context;
  std::vector<std::unique_ptr<tidy::ClangTidyCheck>> Checks;
  ast_matchers::MatchFinder Finder;
};

} // namespace

std::unique_ptr<FeatureModule::ASTListener>
ClangTidyFeatureModule::astListeners() {
  std::lock_guard<std::mutex> Lock(ProviderMu);
  if (!Provider)
    return nullptr;
  return std::make_unique<TidyASTListener>(Provider);
}

void ClangTidyFeatureModule::setProvider(TidyProvider OptionsProvider) {
  std::shared_ptr<const TidyProvider> Replacement;
  if (OptionsProvider)
    Replacement =
        std::make_shared<const TidyProvider>(std::move(OptionsProvider));
  swapProvider(Replacement);
}

void ClangTidyFeatureModule::swapProvider(
    std::shared_ptr<const TidyProvider> &OptionsProvider) {
  std::lock_guard<std::mutex> Lock(ProviderMu);
  Provider.swap(OptionsProvider);
}

} // namespace clangd
} // namespace clang
