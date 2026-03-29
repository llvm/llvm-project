//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_INCLUDEINSERTER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_INCLUDEINSERTER_H

#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/StringSet.h"
#include <optional>

namespace clang {
class Preprocessor;
namespace tidy::utils {

/// Produces fixes to insert specified includes to source files, if not
/// yet present.
///
/// ``IncludeInserter`` can be used in clang-tidy checks in the following way:
/// \code
/// #include "../ClangTidyCheck.h"
/// #include "../utils/IncludeInserter.h"
///
/// namespace clang::tidy {
///
/// class MyCheck : public ClangTidyCheck {
///  public:
///   void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
///                            Preprocessor *ModuleExpanderPP) override {
///     Inserter.registerPreprocessor(PP);
///   }
///
///   void registerMatchers(ast_matchers::MatchFinder* Finder) override { ... }
///
///   void check(
///       const ast_matchers::MatchFinder::MatchResult& Result) override {
///     ...
///     Inserter.createMainFileIncludeInsertion("path/to/Header.h");
///     ...
///   }
///
///  private:
///   utils::IncludeInserter Inserter;
/// };
/// } // namespace clang::tidy
/// \endcode
class IncludeInserter {
public:
  explicit IncludeInserter(bool SelfContainedDiags);

  /// Registers this with the Preprocessor \p PP, must be called before this
  /// class is used.
  void registerPreprocessor(Preprocessor *PP);

  /// Creates a \p Header inclusion directive fixit in the File \p FileID.
  /// When \p Header is enclosed in angle brackets, uses angle brackets in the
  /// inclusion directive, otherwise uses quotes.
  /// Returns ``std::nullopt`` on error or if the inclusion directive already
  /// exists.
  std::optional<FixItHint> createIncludeInsertion(FileID FileID,
                                                  StringRef Header);

  /// Creates a \p Header inclusion directive fixit in the main file.
  /// When \p Header is enclosed in angle brackets, uses angle brackets in the
  /// inclusion directive, otherwise uses quotes.
  /// Returns ``std::nullopt`` on error or if the inclusion directive already
  /// exists.
  std::optional<FixItHint> createMainFileIncludeInsertion(StringRef Header);

private:
  struct InsertInfo {
    SourceLocation InsertionLoc;
    llvm::StringSet<> AlreadyPresentHeaders;
  };

  FileID CurrentFileID;
  llvm::DenseMap<FileID, InsertInfo> InsertInfos;
  const SourceManager *SourceMgr = nullptr;
  const bool SelfContainedDiags;
  friend class IncludeInserterCallback;
};

} // namespace tidy::utils
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_INCLUDEINSERTER_H
