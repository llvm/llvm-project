//===---------- TransformerClangTidyCheck.h - clang-tidy ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_TRANSFORMER_CLANG_TIDY_CHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_TRANSFORMER_CLANG_TIDY_CHECK_H

#include "../ClangTidy.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Refactoring/Transformer.h"
#include <deque>
#include <vector>

namespace clang {
namespace tidy {
namespace utils {

// A base class for defining a ClangTidy check based on a `RewriteRule`.
//
// For example, given a rule `MyCheckAsRewriteRule`, one can define a tidy check
// as follows:
//
// class MyCheck : public TransformerClangTidyCheck {
//  public:
//   MyCheck(StringRef Name, ClangTidyContext *Context)
//       : TransformerClangTidyCheck(MyCheckAsRewriteRule, Name, Context) {}
// };
class TransformerClangTidyCheck : public ClangTidyCheck {
public:
  // All cases in \p R must have a non-null \c Explanation, even though \c
  // Explanation is optional for RewriteRule in general. Because the primary
  // purpose of clang-tidy checks is to provide users with diagnostics, we
  // assume that a missing explanation is a bug.  If no explanation is desired,
  // indicate that explicitly (for example, by passing `text("no explanation")`
  //  to `makeRule` as the `Explanation` argument).
  TransformerClangTidyCheck(tooling::RewriteRule R, StringRef Name,
                            ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) final;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) final;

private:
  tooling::RewriteRule Rule;
};

} // namespace utils
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_TRANSFORMER_CLANG_TIDY_CHECK_H
