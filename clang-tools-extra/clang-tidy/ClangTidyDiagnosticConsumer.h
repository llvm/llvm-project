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
#include "FileExtensionsSet.h"
#include "NoLintDirectiveHandler.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Tooling/Core/Diagnostic.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Regex.h"
#include <optional>

namespace clang {

class ASTContext;
class SourceManager;

namespace tidy {
class CachedGlobList;

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
  std::vector<std::string> EnabledDiagnosticAliases;
};

/// Contains displayed and ignored diagnostic counters for a ClangTidy run.
struct ClangTidyStats {
  unsigned ErrorsDisplayed = 0;
  unsigned ErrorsIgnoredCheckFilter = 0;
  unsigned ErrorsIgnoredNOLINT = 0;
  unsigned ErrorsIgnoredNonUserCode = 0;
  unsigned ErrorsIgnoredLineFilter = 0;

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
                   bool AllowEnablingAnalyzerAlphaCheckers = false,
                   bool EnableModuleHeadersParsing = false);
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
                         StringRef Description,
                         DiagnosticIDs::Level Level = DiagnosticIDs::Warning);

  DiagnosticBuilder diag(StringRef CheckName, StringRef Description,
                         DiagnosticIDs::Level Level = DiagnosticIDs::Warning);

  DiagnosticBuilder diag(const tooling::Diagnostic &Error);

  /// Report any errors to do with reading the configuration using this method.
  DiagnosticBuilder
  configurationDiag(StringRef Message,
                    DiagnosticIDs::Level Level = DiagnosticIDs::Warning);

  /// Check whether a given diagnostic should be suppressed due to the presence
  /// of a "NOLINT" suppression comment.
  /// This is exposed so that other tools that present clang-tidy diagnostics
  /// (such as clangd) can respect the same suppression rules as clang-tidy.
  /// This does not handle suppression of notes following a suppressed
  /// diagnostic; that is left to the caller as it requires maintaining state in
  /// between calls to this function.
  /// If any NOLINT is malformed, e.g. a BEGIN without a subsequent END, output
  /// \param NoLintErrors will return an error about it.
  /// If \param AllowIO is false, the function does not attempt to read source
  /// files from disk which are not already mapped into memory; such files are
  /// treated as not containing a suppression comment.
  /// \param EnableNoLintBlocks controls whether to honor NOLINTBEGIN/NOLINTEND
  /// blocks; if false, only considers line-level disabling.
  bool
  shouldSuppressDiagnostic(DiagnosticsEngine::Level DiagLevel,
                           const Diagnostic &Info,
                           SmallVectorImpl<tooling::Diagnostic> &NoLintErrors,
                           bool AllowIO = true, bool EnableNoLintBlocks = true);

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

  const FileExtensionsSet &getHeaderFileExtensions() const {
    return HeaderFileExtensions;
  }

  const FileExtensionsSet &getImplementationFileExtensions() const {
    return ImplementationFileExtensions;
  }

  /// Returns \c ClangTidyStats containing issued and ignored diagnostic
  /// counters.
  const ClangTidyStats &getStats() const { return Stats; }

  /// Control profile collection in clang-tidy.
  void setEnableProfiling(bool Profile);
  bool getEnableProfiling() const { return Profile; }

  /// Control storage of profile date.
  void setProfileStoragePrefix(StringRef ProfilePrefix);
  std::optional<ClangTidyProfiling::StorageParams>
  getProfileStorageParams() const;

  /// Should be called when starting to process new translation unit.
  void setCurrentBuildDirectory(StringRef BuildDirectory) {
    CurrentBuildDirectory = std::string(BuildDirectory);
  }

  /// Returns build directory of the current translation unit.
  const std::string &getCurrentBuildDirectory() const {
    return CurrentBuildDirectory;
  }

  /// If the experimental alpha checkers from the static analyzer can be
  /// enabled.
  bool canEnableAnalyzerAlphaCheckers() const {
    return AllowEnablingAnalyzerAlphaCheckers;
  }

  // This method determines whether preprocessor-level module header parsing is
  // enabled using the `--experimental-enable-module-headers-parsing` option.
  bool canEnableModuleHeadersParsing() const {
    return EnableModuleHeadersParsing;
  }

  void setSelfContainedDiags(bool Value) { SelfContainedDiags = Value; }

  bool areDiagsSelfContained() const { return SelfContainedDiags; }

  using DiagLevelAndFormatString = std::pair<DiagnosticIDs::Level, std::string>;
  DiagLevelAndFormatString getDiagLevelAndFormatString(unsigned DiagnosticID,
                                                       SourceLocation Loc) {
    return {
        static_cast<DiagnosticIDs::Level>(
            DiagEngine->getDiagnosticLevel(DiagnosticID, Loc)),
        std::string(
            DiagEngine->getDiagnosticIDs()->getDescription(DiagnosticID))};
  }

  void setOptionsCollector(llvm::StringSet<> *Collector) {
    OptionsCollector = Collector;
  }
  llvm::StringSet<> *getOptionsCollector() const { return OptionsCollector; }

private:
  // Writes to Stats.
  friend class ClangTidyDiagnosticConsumer;

  DiagnosticsEngine *DiagEngine = nullptr;
  std::unique_ptr<ClangTidyOptionsProvider> OptionsProvider;

  std::string CurrentFile;
  ClangTidyOptions CurrentOptions;

  std::unique_ptr<CachedGlobList> CheckFilter;
  std::unique_ptr<CachedGlobList> WarningAsErrorFilter;

  FileExtensionsSet HeaderFileExtensions;
  FileExtensionsSet ImplementationFileExtensions;

  LangOptions LangOpts;

  ClangTidyStats Stats;

  std::string CurrentBuildDirectory;

  llvm::DenseMap<unsigned, std::string> CheckNamesByDiagnosticID;

  bool Profile = false;
  std::string ProfilePrefix;

  bool AllowEnablingAnalyzerAlphaCheckers;
  bool EnableModuleHeadersParsing;

  bool SelfContainedDiags = false;

  NoLintDirectiveHandler NoLintHandler;
  llvm::StringSet<> *OptionsCollector = nullptr;
};

/// Gets the Fix attached to \p Diagnostic.
/// If there isn't a Fix attached to the diagnostic and \p AnyFix is true, Check
/// to see if exactly one note has a Fix and return it. Otherwise return
/// nullptr.
const llvm::StringMap<tooling::Replacements> *
getFixIt(const tooling::Diagnostic &Diagnostic, bool AnyFix);

/// A diagnostic consumer that turns each \c Diagnostic into a
/// \c SourceManager-independent \c ClangTidyError.
// FIXME: If we move away from unit-tests, this can be moved to a private
// implementation file.
class ClangTidyDiagnosticConsumer : public DiagnosticConsumer {
public:
  /// \param EnableNolintBlocks Enables diagnostic-disabling inside blocks of
  /// code, delimited by NOLINTBEGIN and NOLINTEND.
  ClangTidyDiagnosticConsumer(ClangTidyContext &Ctx,
                              DiagnosticsEngine *ExternalDiagEngine = nullptr,
                              bool RemoveIncompatibleErrors = true,
                              bool GetFixesFromNotes = false,
                              bool EnableNolintBlocks = true);

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
  void removeDuplicatedDiagnosticsOfAliasCheckers();

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
  bool GetFixesFromNotes;
  bool EnableNolintBlocks;
  std::vector<ClangTidyError> Errors;
  std::unique_ptr<llvm::Regex> HeaderFilter;
  std::unique_ptr<llvm::Regex> ExcludeHeaderFilter;
  bool LastErrorRelatesToUserCode = false;
  bool LastErrorPassesLineFilter = false;
  bool LastErrorWasIgnored = false;
};

} // end namespace tidy
} // end namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGTIDYDIAGNOSTICCONSUMER_H
