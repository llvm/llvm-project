//===---------- IncludeInserter.h - clang-tidy ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_INCLUDEINSERTER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_INCLUDEINSERTER_H

#include "IncludeSorter.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/StringSet.h"
#include <memory>

namespace clang {
class Preprocessor;
namespace tidy {
namespace utils {

/// Produces fixes to insert specified includes to source files, if not
/// yet present.
///
/// ``IncludeInserter`` can be used in clang-tidy checks in the following way:
/// \code
/// #include "../ClangTidyCheck.h"
/// #include "../utils/IncludeInserter.h"
///
/// namespace clang {
/// namespace tidy {
///
/// class MyCheck : public ClangTidyCheck {
///  public:
///   void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
///                            Preprocessor *ModuleExpanderPP) override {
///     Inserter.registerPreprocessor();
///   }
///
///   void registerMatchers(ast_matchers::MatchFinder* Finder) override { ... }
///
///   void check(
///       const ast_matchers::MatchFinder::MatchResult& Result) override {
///     ...
///     Inserter.createMainFileIncludeInsertion("path/to/Header.h",
///                                             /*IsAngled=*/false);
///     ...
///   }
///
///  private:
///   utils::IncludeInserter Inserter{utils::IncludeSorter::IS_Google};
/// };
/// } // namespace tidy
/// } // namespace clang
/// \endcode
class IncludeInserter {
public:
  /// Initializes the IncludeInserter using the IncludeStyle \p Style.
  /// In most cases the \p Style will be retrieved from the ClangTidyOptions
  /// using \code
  ///   Options.getLocalOrGlobal("IncludeStyle", <DefaultStyle>)
  /// \endcode
  explicit IncludeInserter(IncludeSorter::IncludeStyle Style);

  /// Registers this with the Preprocessor \p PP, must be called before this
  /// class is used.
  void registerPreprocessor(Preprocessor *PP);

  /// Creates a \p Header inclusion directive fixit in the File \p FileID.
  /// Returns ``llvm::None`` on error or if the inclusion directive already
  /// exists.
  /// FIXME: This should be removed once the clients are migrated to the
  /// overload without the ``IsAngled`` parameter.
  llvm::Optional<FixItHint>
  createIncludeInsertion(FileID FileID, llvm::StringRef Header, bool IsAngled);

  /// Creates a \p Header inclusion directive fixit in the File \p FileID.
  /// When \p Header is enclosed in angle brackets, uses angle brackets in the
  /// inclusion directive, otherwise uses quotes.
  /// Returns ``llvm::None`` on error or if the inclusion directive already
  /// exists.
  llvm::Optional<FixItHint> createIncludeInsertion(FileID FileID,
                                                   llvm::StringRef Header);

  /// Creates a \p Header inclusion directive fixit in the main file.
  /// Returns``llvm::None`` on error or if the inclusion directive already
  /// exists.
  /// FIXME: This should be removed once the clients are migrated to the
  /// overload without the ``IsAngled`` parameter.
  llvm::Optional<FixItHint>
  createMainFileIncludeInsertion(llvm::StringRef Header, bool IsAngled);

  /// Creates a \p Header inclusion directive fixit in the main file.
  /// When \p Header is enclosed in angle brackets, uses angle brackets in the
  /// inclusion directive, otherwise uses quotes.
  /// Returns``llvm::None`` on error or if the inclusion directive already
  /// exists.
  llvm::Optional<FixItHint>
  createMainFileIncludeInsertion(llvm::StringRef Header);

  IncludeSorter::IncludeStyle getStyle() const { return Style; }

private:
  void addInclude(StringRef FileName, bool IsAngled,
                  SourceLocation HashLocation, SourceLocation EndLocation);

  IncludeSorter &getOrCreate(FileID FileID);

  llvm::DenseMap<FileID, std::unique_ptr<IncludeSorter>> IncludeSorterByFile;
  llvm::DenseMap<FileID, llvm::StringSet<>> InsertedHeaders;
  const SourceManager *SourceMgr{nullptr};
  const IncludeSorter::IncludeStyle Style;
  friend class IncludeInserterCallback;
};

} // namespace utils
} // namespace tidy
} // namespace clang
#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_INCLUDEINSERTER_H
