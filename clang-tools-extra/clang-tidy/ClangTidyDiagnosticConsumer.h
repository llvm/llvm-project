//===--- ClangTidyDiagnosticConsumer.h - clang-tidy -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGTIDYDIAGNOSTICCONSUMER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGTIDYDIAGNOSTICCONSUMER_H

#include "ClangTidyOptions.h"
#include "ClangTidyProfiling.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Tooling/Core/Diagnostic.h"
#include "clang/Tooling/Refactoring.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Timer.h"

namespace clang {

class ASTContext;
class CompilerInstance;
namespace ast_matchers {
class MatchFinder;
}
namespace tooling {
class CompilationDatabase;
}

namespace tidy {

/// A detected error complete with information to display diagnostic and
/// automatic fix.
///
/// This is used as an intermediate format to transport Diagnostics without a
/// dependency on a SourceManager.
///
/// FIXME: Make Diagnostics flexible enough to support this directly.
struct ClangTidyError : tooling::Diagnostic {
  ClangTidyError(StringRef CheckName, Level DiagLevel, StringRef BuildDirectory,
                 bool IsWarningAsError);

  bool IsWarningAsError;
};

/// Contains displayed and ignored diagnostic counters for a ClangTidy
/// run.
struct ClangTidyStats {
  ClangTidyStats()
      : ErrorsDisplayed(0), ErrorsIgnoredCheckFilter(0), ErrorsIgnoredNOLINT(0),
        ErrorsIgnoredNonUserCode(0), ErrorsIgnoredLineFilter(0) {}

  unsigned ErrorsDisplayed;
  unsigned ErrorsIgnoredCheckFilter;
  unsigned ErrorsIgnoredNOLINT;
  unsigned ErrorsIgnoredNonUserCode;
  unsigned ErrorsIgnoredLineFilter;

  unsigned errorsIgnored() const {
    return ErrorsIgnoredNOLINT + ErrorsIgnoredCheckFilter +
           ErrorsIgnoredNonUserCode + ErrorsIgnoredLineFilter;
  }
};

/// Every \c ClangTidyCheck reports errors through a \c DiagnosticsEngine
/// provided by this context.
///
/// A \c ClangTidyCheck always has access to the active context to report
/// warnings like:
/// \code
/// Context->Diag(Loc, "Single-argument constructors must be explicit")
///     << FixItHint::CreateInsertion(Loc, "explicit ");
/// \endcode
class ClangTidyContext {
public:
  /// Initializes \c ClangTidyContext instance.
  ClangTidyContext(std::unique_ptr<ClangTidyOptionsProvider> OptionsProvider,
                   bool AllowEnablingAnalyzerAlphaCheckers = false);
  /// Sets the DiagnosticsEngine that diag() will emit diagnostics to.
  // FIXME: this is required initialization, and should be a constructor param.
  // Fix the context -> diag engine -> consumer -> context initialization cycle.
  void setDiagnosticsEngine(DiagnosticsEngine *DiagEngine) {
    this->DiagEngine = DiagEngine;
  }

  ~ClangTidyContext();

  /// Report any errors detected using this method.
  ///
  /// This is still under heavy development and will likely change towards using
  /// tablegen'd diagnostic IDs.
  /// FIXME: Figure out a way to manage ID spaces.
  DiagnosticBuilder diag(StringRef CheckName, SourceLocation Loc,
                         StringRef Message,
                         DiagnosticIDs::Level Level = DiagnosticIDs::Warning);

  /// Sets the \c SourceManager of the used \c DiagnosticsEngine.
  ///
  /// This is called from the \c ClangTidyCheck base class.
  void setSourceManager(SourceManager *SourceMgr);

  /// Should be called when starting to process new translation unit.
  void setCurrentFile(StringRef File);

  /// Returns the main file name of the current translation unit.
  StringRef getCurrentFile() const { return CurrentFile; }

  /// Sets ASTContext for the current translation unit.
  void setASTContext(ASTContext *Context);

  /// Gets the language options from the AST context.
  const LangOptions &getLangOpts() const { return LangOpts; }

  /// Returns the name of the clang-tidy check which produced this
  /// diagnostic ID.
  std::string getCheckName(unsigned DiagnosticID) const;

  /// Returns \c true if the check is enabled for the \c CurrentFile.
  ///
  /// The \c CurrentFile can be changed using \c setCurrentFile.
  bool isCheckEnabled(StringRef CheckName) const;

  /// Returns \c true if the check should be upgraded to error for the
  /// \c CurrentFile.
  bool treatAsError(StringRef CheckName) const;

  /// Returns global options.
  const ClangTidyGlobalOptions &getGlobalOptions() const;

  /// Returns options for \c CurrentFile.
  ///
  /// The \c CurrentFile can be changed using \c setCurrentFile.
  const ClangTidyOptions &getOptions() const;

  /// Returns options for \c File. Does not change or depend on
  /// \c CurrentFile.
  ClangTidyOptions getOptionsForFile(StringRef File) const;

  /// Returns \c ClangTidyStats containing issued and ignored diagnostic
  /// counters.
  const ClangTidyStats &getStats() const { return Stats; }

  /// Control profile collection in clang-tidy.
  void setEnableProfiling(bool Profile);
  bool getEnableProfiling() const { return Profile; }

  /// Control storage of profile date.
  void setProfileStoragePrefix(StringRef ProfilePrefix);
  llvm::Optional<ClangTidyProfiling::StorageParams>
  getProfileStorageParams() const;

  /// Should be called when starting to process new translation unit.
  void setCurrentBuildDirectory(StringRef BuildDirectory) {
    CurrentBuildDirectory = std::string(BuildDirectory);
  }

  /// Returns build directory of the current translation unit.
  const std::string &getCurrentBuildDirectory() {
    return CurrentBuildDirectory;
  }

  /// If the experimental alpha checkers from the static analyzer can be
  /// enabled.
  bool canEnableAnalyzerAlphaCheckers() const {
    return AllowEnablingAnalyzerAlphaCheckers;
  }

  using DiagLevelAndFormatString = std::pair<DiagnosticIDs::Level, std::string>;
  DiagLevelAndFormatString getDiagLevelAndFormatString(unsigned DiagnosticID,
                                                       SourceLocation Loc) {
    return DiagLevelAndFormatString(
        static_cast<DiagnosticIDs::Level>(
            DiagEngine->getDiagnosticLevel(DiagnosticID, Loc)),
        std::string(
            DiagEngine->getDiagnosticIDs()->getDescription(DiagnosticID)));
  }

private:
  // Writes to Stats.
  friend class ClangTidyDiagnosticConsumer;

  DiagnosticsEngine *DiagEngine;
  std::unique_ptr<ClangTidyOptionsProvider> OptionsProvider;

  std::string CurrentFile;
  ClangTidyOptions CurrentOptions;
  class CachedGlobList;
  std::unique_ptr<CachedGlobList> CheckFilter;
  std::unique_ptr<CachedGlobList> WarningAsErrorFilter;

  LangOptions LangOpts;

  ClangTidyStats Stats;

  std::string CurrentBuildDirectory;

  llvm::DenseMap<unsigned, std::string> CheckNamesByDiagnosticID;

  bool Profile;
  std::string ProfilePrefix;

  bool AllowEnablingAnalyzerAlphaCheckers;
};

/// Check whether a given diagnostic should be suppressed due to the presence
/// of a "NOLINT" suppression comment.
/// This is exposed so that other tools that present clang-tidy diagnostics
/// (such as clangd) can respect the same suppression rules as clang-tidy.
/// This does not handle suppression of notes following a suppressed diagnostic;
/// that is left to the caller is it requires maintaining state in between calls
/// to this function.
/// The `CheckMacroExpansion` parameter determines whether the function should
/// handle the case where the diagnostic is inside a macro expansion. A degree
/// of control over this is needed because handling this case can require
/// examining source files other than the one in which the diagnostic is
/// located, and in some use cases we cannot rely on such other files being
/// mapped in the SourceMapper.
bool shouldSuppressDiagnostic(DiagnosticsEngine::Level DiagLevel,
                              const Diagnostic &Info, ClangTidyContext &Context,
                              bool CheckMacroExpansion = true);

/// A diagnostic consumer that turns each \c Diagnostic into a
/// \c SourceManager-independent \c ClangTidyError.
//
// FIXME: If we move away from unit-tests, this can be moved to a private
// implementation file.
class ClangTidyDiagnosticConsumer : public DiagnosticConsumer {
public:
  ClangTidyDiagnosticConsumer(ClangTidyContext &Ctx,
                              DiagnosticsEngine *ExternalDiagEngine = nullptr,
                              bool RemoveIncompatibleErrors = true);

  // FIXME: The concept of converting between FixItHints and Replacements is
  // more generic and should be pulled out into a more useful Diagnostics
  // library.
  void HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                        const Diagnostic &Info) override;

  // Retrieve the diagnostics that were captured.
  std::vector<ClangTidyError> take();

private:
  void finalizeLastError();
  void removeIncompatibleErrors();

  /// Returns the \c HeaderFilter constructed for the options set in the
  /// context.
  llvm::Regex *getHeaderFilter();

  /// Updates \c LastErrorRelatesToUserCode and LastErrorPassesLineFilter
  /// according to the diagnostic \p Location.
  void checkFilters(SourceLocation Location, const SourceManager &Sources);
  bool passesLineFilter(StringRef FileName, unsigned LineNumber) const;

  void forwardDiagnostic(const Diagnostic &Info);

  ClangTidyContext &Context;
  DiagnosticsEngine *ExternalDiagEngine;
  bool RemoveIncompatibleErrors;
  std::vector<ClangTidyError> Errors;
  std::unique_ptr<llvm::Regex> HeaderFilter;
  bool LastErrorRelatesToUserCode;
  bool LastErrorPassesLineFilter;
  bool LastErrorWasIgnored;
};

} // end namespace tidy
} // end namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGTIDYDIAGNOSTICCONSUMER_H
