//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USEINITSTATEMENTCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USEINITSTATEMENTCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::modernize {

/// FIXME: Write a short description.
///
/// For the user-facing documentation see:
/// https://clang.llvm.org/extra/clang-tidy/checks/modernize/use-init-statement.html
class UseInitStatementCheck : public ClangTidyCheck {
public:
  UseInitStatementCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus17;
  }

private:
  const DeclStmt* findPreviousDeclStmt(const Stmt *CurrentStmt, const VarDecl *TargetVar, 
                                      ASTContext *Context);
  bool isVariableUsedInStmt(const VarDecl *VD, const Stmt *S);
  bool isVariableUsedAfterStmt(const VarDecl *VD, const Stmt *Stmt, ASTContext *Context);
};

} // namespace clang::tidy::modernize

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_USEINITSTATEMENTCHECK_H
