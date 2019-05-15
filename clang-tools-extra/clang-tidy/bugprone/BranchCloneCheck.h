//===--- BranchCloneCheck.h - clang-tidy ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_BRANCHCLONECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_BRANCHCLONECHECK_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace bugprone {

/// A check for detecting if/else if/else chains where two or more branches are
/// Type I clones of each other (that is, they contain identical code), for
/// detecting switch statements where two or more consecutive branches are
/// Type I clones of each other, and for detecting conditional operators where
/// the true and false expressions are Type I clones of each other.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone-branch-clone.html
class BranchCloneCheck : public ClangTidyCheck {
public:
  BranchCloneCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace bugprone
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_BRANCHCLONECHECK_H
