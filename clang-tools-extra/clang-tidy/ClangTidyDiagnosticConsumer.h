//===--- ClangTidyDiagnosticConsumer.h - clang-tidy -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGTIDYDIAGNOSTICCONSUMER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGTIDYDIAGNOSTICCONSUMER_H

#include "ClangTidyOptions.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Tooling/Refactoring.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Regex.h"
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

/// \brief A message from a clang-tidy check.
///
/// Note that this is independent of a \c SourceManager.
struct ClangTidyMessage {
  ClangTidyMessage(StringRef Message = "");
  ClangTidyMessage(StringRef Message, const SourceManager &Sources,
                   SourceLocation Loc);
  std::string Message;
  std::string FilePath;
  unsigned FileOffset;
};

/// \brief A detected error complete with information to display diagnostic and
/// automatic fix.
///
/// This is used as an intermediate format to transport Diagnostics without a
/// dependency on a SourceManager.
///
/// FIXME: Make Diagnostics flexible enough to support this directly.
struct ClangTidyError {
  enum Level {
    Warning = DiagnosticsEngine::Warning,
    Error = DiagnosticsEngine::Error
  };

  ClangTidyError(StringRef CheckName, Level DiagLevel, bool IsWarningAsError,
                 StringRef BuildDirectory);

  std::string CheckName;
  ClangTidyMessage Message;
  // Fixes grouped by file path.
  llvm::StringMap<tooling::Replacements> Fix;
  SmallVector<ClangTidyMessage, 1> Notes;

  // A build directory of the diagnostic source file.
  //
  // It's an absolute path which is `directory` field of the source file in
  // compilation database. If users don't specify the compilation database
  // directory, it is the current directory where clang-tidy runs.
  //
  // Note: it is empty in unittest.
  std::string BuildDirectory;

  Level DiagLevel;
  bool IsWarningAsError;
};

/// \brief Read-only set of strings represented as a list of positive and
/// negative globs. Positive globs add all matched strings to the set, negative
/// globs remove them in the order of appearance in the list.
class GlobList {
public:
  /// \brief \p GlobList is a comma-separated list of globs (only '*'
  /// metacharacter is supported) with optional '-' prefix to denote exclusion.
  GlobList(StringRef Globs);

  /// \brief Returns \c true if the pattern matches \p S. The result is the last
  /// matching glob's Positive flag.
  bool contains(StringRef S) { return contains(S, false); }

private:
  bool contains(StringRef S, bool Contains);

  bool Positive;
  llvm::Regex Regex;
  std::unique_ptr<GlobList> NextGlob;
};

/// \brief Contains displayed and ignored diagnostic counters for a ClangTidy
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

/// \brief Container for clang-tidy profiling data.
struct ProfileData {
  llvm::StringMap<llvm::TimeRecord> Records;
};

/// \brief Every \c ClangTidyCheck reports errors through a \c DiagnosticsEngine
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
  /// \brief Initializes \c ClangTidyContext instance.
  ClangTidyContext(std::unique_ptr<ClangTidyOptionsProvider> OptionsProvider);

  /// \brief Report any errors detected using this method.
  ///
  /// This is still under heavy development and will likely change towards using
  /// tablegen'd diagnostic IDs.
  /// FIXME: Figure out a way to manage ID spaces.
  DiagnosticBuilder diag(StringRef CheckName, SourceLocation Loc,
                         StringRef Message,
                         DiagnosticIDs::Level Level = DiagnosticIDs::Warning);

  /// \brief Sets the \c SourceManager of the used \c DiagnosticsEngine.
  ///
  /// This is called from the \c ClangTidyCheck base class.
  void setSourceManager(SourceManager *SourceMgr);

  /// \brief Should be called when starting to process new translation unit.
  void setCurrentFile(StringRef File);

  /// \brief Returns the main file name of the current translation unit.
  StringRef getCurrentFile() const { return CurrentFile; }

  /// \brief Sets ASTContext for the current translation unit.
  void setASTContext(ASTContext *Context);

  /// \brief Gets the language options from the AST context.
  const LangOptions &getLangOpts() const { return LangOpts; }

  /// \brief Returns the name of the clang-tidy check which produced this
  /// diagnostic ID.
  StringRef getCheckName(unsigned DiagnosticID) const;

  /// \brief Returns check filter for the \c CurrentFile.
  ///
  /// The \c CurrentFile can be changed using \c setCurrentFile.
  GlobList &getChecksFilter();

  /// \brief Returns check filter for the \c CurrentFile which
  /// selects checks for upgrade to error.
  GlobList &getWarningAsErrorFilter();

  /// \brief Returns global options.
  const ClangTidyGlobalOptions &getGlobalOptions() const;

  /// \brief Returns options for \c CurrentFile.
  ///
  /// The \c CurrentFile can be changed using \c setCurrentFile.
  const ClangTidyOptions &getOptions() const;

  /// \brief Returns options for \c File. Does not change or depend on
  /// \c CurrentFile.
  ClangTidyOptions getOptionsForFile(StringRef File) const;

  /// \brief Returns \c ClangTidyStats containing issued and ignored diagnostic
  /// counters.
  const ClangTidyStats &getStats() const { return Stats; }

  /// \brief Returns all collected errors.
  const std::vector<ClangTidyError> &getErrors() const { return Errors; }

  /// \brief Clears collected errors.
  void clearErrors() { Errors.clear(); }

  /// \brief Set the output struct for profile data.
  ///
  /// Setting a non-null pointer here will enable profile collection in
  /// clang-tidy.
  void setCheckProfileData(ProfileData *Profile);
  ProfileData *getCheckProfileData() const { return Profile; }

  /// \brief Should be called when starting to process new translation unit.
  void setCurrentBuildDirectory(StringRef BuildDirectory) {
    CurrentBuildDirectory = BuildDirectory;
  }

  /// \brief Returns build directory of the current translation unit.
  const std::string &getCurrentBuildDirectory() {
    return CurrentBuildDirectory;
  }

private:
  // Calls setDiagnosticsEngine() and storeError().
  friend class ClangTidyDiagnosticConsumer;
  friend class ClangTidyPluginAction;

  /// \brief Sets the \c DiagnosticsEngine so that Diagnostics can be generated
  /// correctly.
  void setDiagnosticsEngine(DiagnosticsEngine *Engine);

  /// \brief Store an \p Error.
  void storeError(const ClangTidyError &Error);

  std::vector<ClangTidyError> Errors;
  DiagnosticsEngine *DiagEngine;
  std::unique_ptr<ClangTidyOptionsProvider> OptionsProvider;

  std::string CurrentFile;
  ClangTidyOptions CurrentOptions;
  std::unique_ptr<GlobList> CheckFilter;
  std::unique_ptr<GlobList> WarningAsErrorFilter;

  LangOptions LangOpts;

  ClangTidyStats Stats;

  std::string CurrentBuildDirectory;

  llvm::DenseMap<unsigned, std::string> CheckNamesByDiagnosticID;

  ProfileData *Profile;
};

/// \brief A diagnostic consumer that turns each \c Diagnostic into a
/// \c SourceManager-independent \c ClangTidyError.
//
// FIXME: If we move away from unit-tests, this can be moved to a private
// implementation file.
class ClangTidyDiagnosticConsumer : public DiagnosticConsumer {
public:
  ClangTidyDiagnosticConsumer(ClangTidyContext &Ctx);

  // FIXME: The concept of converting between FixItHints and Replacements is
  // more generic and should be pulled out into a more useful Diagnostics
  // library.
  void HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                        const Diagnostic &Info) override;

  /// \brief Flushes the internal diagnostics buffer to the ClangTidyContext.
  void finish() override;

private:
  void finalizeLastError();

  void removeIncompatibleErrors(SmallVectorImpl<ClangTidyError> &Errors) const;

  /// \brief Returns the \c HeaderFilter constructed for the options set in the
  /// context.
  llvm::Regex *getHeaderFilter();

  /// \brief Updates \c LastErrorRelatesToUserCode and LastErrorPassesLineFilter
  /// according to the diagnostic \p Location.
  void checkFilters(SourceLocation Location);
  bool passesLineFilter(StringRef FileName, unsigned LineNumber) const;

  ClangTidyContext &Context;
  std::unique_ptr<DiagnosticsEngine> Diags;
  SmallVector<ClangTidyError, 8> Errors;
  std::unique_ptr<llvm::Regex> HeaderFilter;
  bool LastErrorRelatesToUserCode;
  bool LastErrorPassesLineFilter;
};

} // end namespace tidy
} // end namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGTIDYDIAGNOSTICCONSUMER_H
