//===--- PrimitiveVarDeclRequirement.cpp - Clang refactoring library ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "clang/Tooling/Refactoring/RefactoringActionRuleRequirements.h"
#include "clang/Tooling/Refactoring/VarInits/PrimitiveVarDecl.h"
#include <optional>

using namespace clang;
using namespace tooling;

Expected<VarDecl *>
PrimitiveVarDeclRequirement::evaluate(RefactoringRuleContext &Context) const {
  Expected<SourceLocation> Location =
      SourceLocationRequirement::evaluate(Context);
  if (!Location)
    return Location.takeError();

  DeclRefExpr *VariableReference = getDeclRefExprFromSourceLocation(
      Context.getASTContext(), Location.get());

  if (!VariableReference)
    return Context.createDiagnosticError(
        Location.get(), diag::err_refactor_no_vardecl);

  VarDecl *Variable = VariableReference->getDecl()->
                      getPotentiallyDecomposedVarDecl();
  return std::move(Variable);
}
