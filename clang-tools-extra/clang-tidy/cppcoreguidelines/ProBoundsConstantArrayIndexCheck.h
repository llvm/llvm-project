//===--- ProBoundsConstantArrayIndexCheck.h - clang-tidy---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_PRO_BOUNDS_CONSTANT_ARRAY_INDEX_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_PRO_BOUNDS_CONSTANT_ARRAY_INDEX_H

#include "../ClangTidy.h"
#include "../utils/IncludeInserter.h"

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

/// This checks that all array subscriptions on static arrays and std::arrays
/// have a constant index and are within bounds
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/cppcoreguidelines-pro-bounds-constant-array-index.html
class ProBoundsConstantArrayIndexCheck : public ClangTidyCheck {
  const std::string GslHeader;
  const utils::IncludeSorter::IncludeStyle IncludeStyle;
  std::unique_ptr<utils::IncludeInserter> Inserter;

public:
  ProBoundsConstantArrayIndexCheck(StringRef Name, ClangTidyContext *Context);
  void registerPPCallbacks(CompilerInstance &Compiler) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_PRO_BOUNDS_CONSTANT_ARRAY_INDEX_H
