//===--- ASTStatementRequirements.cpp - Clang refactoring library ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Refactoring/RefactoringActionRuleRequirements.h"
#include "clang/Tooling/Refactoring/ASTStatement.h"

using namespace clang;
using namespace tooling;

Expected<Stmt *>
ASTStatementRequirement::evaluate(RefactoringRuleContext &Context) const {
  Expected<SourceLocation> Location =
      SourceLocationRequirement::evaluate(Context);
  if (!Location)
    return Location.takeError();

  Stmt *Statement =
      findOuterStmt(Context.getASTContext(), *Location);
  if (Statement == nullptr)
    return Context.createDiagnosticError(
        *Location, diag::err_refactor_location_no_statement);
  return std::move(Statement);
}
