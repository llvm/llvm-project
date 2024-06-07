//===--- DeleteStatementRule.cpp - Clang refactoring library --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implements the "delete-statement" refactoring rule that can delete stmts
///
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Refactoring/ASTStatement.h"
#include "clang/Tooling/Refactoring/Delete/DeleteStatementRule.h"
#include "clang/AST/ASTContext.h"
#include "clang/Rewrite/Core/Rewriter.h"

using namespace clang;
using namespace tooling;

Expected<DeleteStatementRule>
DeleteStatementRule::initiate(RefactoringRuleContext &Context, Stmt *Stmt) {
  return DeleteStatementRule(std::move(Stmt));
}

const RefactoringDescriptor &DeleteStatementRule::describe() {
  static const RefactoringDescriptor Descriptor = {
      "delete-statement",
      "Delete Statement",
      "Deletes stmts from code",
  };
  return Descriptor;
}

Expected<AtomicChanges>
DeleteStatementRule::createSourceReplacements(RefactoringRuleContext &Context) {
  // Compute the source range of the code that should be deleted.
  SourceRange DeleteRange(Statement->getBeginLoc(),
                             Statement->getEndLoc());

  ASTContext &AST = Context.getASTContext();
  SourceManager &SM = AST.getSourceManager();
  const LangOptions &LangOpts = AST.getLangOpts();
  Rewriter DeleteCodeRewriter(SM, LangOpts);

  PrintingPolicy PP = AST.getPrintingPolicy();
  PP.SuppressStrongLifetime = true;
  PP.SuppressLifetimeQualifiers = true;
  PP.SuppressUnwrittenScope = true;

  AtomicChange Change(SM, Statement->getBeginLoc());
  // Create the replacement for deleting statement
  {
    auto Err = Change.replace(
        SM, CharSourceRange::getTokenRange(DeleteRange), "");
    if (Err)
      return std::move(Err);
  }

  return AtomicChanges{std::move(Change)};
}
