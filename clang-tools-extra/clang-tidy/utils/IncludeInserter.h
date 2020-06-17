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
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/PPCallbacks.h"
#include <memory>
#include <string>

namespace clang {
namespace tidy {
namespace utils {

/// Produces fixes to insert specified includes to source files, if not
/// yet present.
///
/// ``IncludeInserter`` can be used in clang-tidy checks in the following way:
/// \code
/// #include "../utils/IncludeInserter.h"
/// #include "clang/Frontend/CompilerInstance.h"
///
/// class MyCheck : public ClangTidyCheck {
///  public:
///   void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
///                            Preprocessor *ModuleExpanderPP) override {
///     Inserter = std::make_unique<IncludeInserter>(
///         SM, getLangOpts(), utils::IncludeSorter::IS_Google);
///     PP->addPPCallbacks(Inserter->CreatePPCallbacks());
///   }
///
///   void registerMatchers(ast_matchers::MatchFinder* Finder) override { ... }
///
///   void check(
///       const ast_matchers::MatchFinder::MatchResult& Result) override {
///     ...
///     Inserter->CreateIncludeInsertion(
///         Result.SourceManager->getMainFileID(), "path/to/Header.h",
///         /*IsAngled=*/false);
///     ...
///   }
///
///  private:
///   std::unique_ptr<clang::tidy::utils::IncludeInserter> Inserter;
/// };
/// \endcode
class IncludeInserter {
public:
  IncludeInserter(const SourceManager &SourceMgr, const LangOptions &LangOpts,
                  IncludeSorter::IncludeStyle Style);
  ~IncludeInserter();

  /// Create ``PPCallbacks`` for registration with the compiler's preprocessor.
  std::unique_ptr<PPCallbacks> CreatePPCallbacks();

  /// Creates a \p Header inclusion directive fixit. Returns ``llvm::None`` on
  /// error or if inclusion directive already exists.
  llvm::Optional<FixItHint>
  CreateIncludeInsertion(FileID FileID, llvm::StringRef Header, bool IsAngled);

private:
  void AddInclude(StringRef FileName, bool IsAngled,
                  SourceLocation HashLocation, SourceLocation EndLocation);

  IncludeSorter &getOrCreate(FileID FileID);

  llvm::DenseMap<FileID, std::unique_ptr<IncludeSorter>> IncludeSorterByFile;
  llvm::DenseMap<FileID, std::set<std::string>> InsertedHeaders;
  const SourceManager &SourceMgr;
  const LangOptions &LangOpts;
  const IncludeSorter::IncludeStyle Style;
  friend class IncludeInserterCallback;
};

} // namespace utils
} // namespace tidy
} // namespace clang
#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_INCLUDEINSERTER_H
