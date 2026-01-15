//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_INCONSISTENTIFELSEBRACESCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_INCONSISTENTIFELSEBRACESCHECK_H

#include "../ClangTidyCheck.h"
#include "clang/AST/ASTTypeTraits.h"
#include <optional>

namespace clang::tidy::readability {

/// Detects `if`/`else` statements where one branch uses braces and the other
/// does not.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/readability/inconsistent-ifelse-braces.html
class InconsistentIfElseBracesCheck : public ClangTidyCheck {
public:
  InconsistentIfElseBracesCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  std::optional<TraversalKind> getCheckTraversalKind() const override {
    return TK_IgnoreUnlessSpelledInSource;
  }

private:
  void checkIfStmt(const ast_matchers::MatchFinder::MatchResult &Result,
                   const IfStmt *If);
  void emitDiagnostic(const ast_matchers::MatchFinder::MatchResult &Result,
                      const Stmt *S, SourceLocation StartLoc,
                      SourceLocation EndLocHint = {});
};

} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_INCONSISTENTIFELSEBRACESCHECK_H
