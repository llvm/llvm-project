//===--- ClangTidy.h - clang-tidy -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGTIDY_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGTIDY_H

#include "ClangTidyDiagnosticConsumer.h"
#include "ClangTidyOptions.h"
#include <memory>
#include <vector>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace clang {

class ASTConsumer;
class CompilerInstance;
namespace tooling {
class CompilationDatabase;
} // namespace tooling

namespace tidy {

class ClangTidyCheckFactories;

class ClangTidyASTConsumerFactory {
public:
  ClangTidyASTConsumerFactory(
      ClangTidyContext &Context,
      IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> OverlayFS = nullptr);

  /// Returns an ASTConsumer that runs the specified clang-tidy checks.
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &Compiler, StringRef File);

  /// Get the list of enabled checks.
  std::vector<std::string> getCheckNames();

  /// Get the union of options from all checks.
  ClangTidyOptions::OptionMap getCheckOptions();

private:
  ClangTidyContext &Context;
  IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> OverlayFS;
  std::unique_ptr<ClangTidyCheckFactories> CheckFactories;
};

/// Fills the list of check names that are enabled when the provided
/// filters are applied.
std::vector<std::string> getCheckNames(const ClangTidyOptions &Options,
                                       bool AllowEnablingAnalyzerAlphaCheckers);

/// Returns the effective check-specific options.
///
/// The method configures ClangTidy with the specified \p Options and collects
/// effective options from all created checks. The returned set of options
/// includes default check-specific options for all keys not overridden by \p
/// Options.
ClangTidyOptions::OptionMap
getCheckOptions(const ClangTidyOptions &Options,
                bool AllowEnablingAnalyzerAlphaCheckers);

/// Run a set of clang-tidy checks on a set of files.
///
/// \param EnableCheckProfile If provided, it enables check profile collection
/// in MatchFinder, and will contain the result of the profile.
/// \param StoreCheckProfile If provided, and EnableCheckProfile is true,
/// the profile will not be output to stderr, but will instead be stored
/// as a JSON file in the specified directory.
std::vector<ClangTidyError>
runClangTidy(clang::tidy::ClangTidyContext &Context,
             const tooling::CompilationDatabase &Compilations,
             ArrayRef<std::string> InputFiles,
             llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> BaseFS,
             bool EnableCheckProfile = false,
             llvm::StringRef StoreCheckProfile = StringRef());

// FIXME: This interface will need to be significantly extended to be useful.
// FIXME: Implement confidence levels for displaying/fixing errors.
//
/// Displays the found \p Errors to the users. If \p Fix is true, \p
/// Errors containing fixes are automatically applied and reformatted. If no
/// clang-format configuration file is found, the given \P FormatStyle is used.
void handleErrors(llvm::ArrayRef<ClangTidyError> Errors,
                  ClangTidyContext &Context, bool Fix,
                  unsigned &WarningsAsErrorsCount,
                  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> BaseFS);

/// Serializes replacements into YAML and writes them to the specified
/// output stream.
void exportReplacements(StringRef MainFilePath,
                        const std::vector<ClangTidyError> &Errors,
                        raw_ostream &OS);

} // end namespace tidy
} // end namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGTIDY_H
