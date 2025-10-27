//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_INCONSISTENTIFELSEBRACESCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_INCONSISTENTIFELSEBRACESCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::bugprone {

/// Detects `if`/`else` statements where one branch uses braces and the other
/// does not.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/inconsistent-ifelse-braces.html
class InconsistentIfelseBracesCheck : public ClangTidyCheck {
public:
  InconsistentIfelseBracesCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  void checkIfStmt(const ast_matchers::MatchFinder::MatchResult &Result,
                   const IfStmt *If);
  void checkStmt(const ast_matchers::MatchFinder::MatchResult &Result,
                 const Stmt *S, SourceLocation StartLoc,
                 SourceLocation EndLocHint = SourceLocation());
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_INCONSISTENTIFELSEBRACESCHECK_H
