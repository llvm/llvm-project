//===--- NoRecursionCheck.h - clang-tidy ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_NORECURSIONCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_NORECURSIONCHECK_H

#include "../ClangTidyCheck.h"

namespace clang {

class CallGraphNode;

namespace tidy {
namespace misc {

/// Finds strongly connected functions (by analyzing call graph for SCC's
/// that are loops), diagnoses each function in the cycle,
/// and displays one example of possible call graph loop (recursion).
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/misc-no-recursion.html
class NoRecursionCheck : public ClangTidyCheck {
public:
  NoRecursionCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  void handleSCC(ArrayRef<CallGraphNode *> SCC);
};

} // namespace misc
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_NORECURSIONCHECK_H
