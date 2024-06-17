//===--- PrimitivesInit.cpp - Clang refactoring library -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Refactoring/VarInits/PrimitivesInit.h"
#include "clang/AST/ASTContext.h"

using namespace clang;
using namespace tooling;

static inline bool VariableHasPrimitiveType(VarDecl *Variable) {
  return Variable->getType()->isCharType()    ||
         Variable->getType()->isIntegerType() ||
         Variable->getType()->isFloatingType();
}

Expected<PrimitivesInit>
PrimitivesInit::initiate(RefactoringRuleContext &Context,
                         VarDecl *Variable) {
  // Checks whether provided VarDecl is valid
  if (!Variable->isLocalVarDecl()) {
    return Context.createDiagnosticError(
        diag::err_refactor_global_variable_init);
  }

  if (!VariableHasPrimitiveType(Variable)) {
    return Context.createDiagnosticError(
        diag::err_refactor_non_primitive_variable);
  }

  if (Variable->hasInit()) {
    return Context.createDiagnosticError(
        diag::err_refactor_initialized_variable);
  }

  return PrimitivesInit(std::move(Variable));
}
const RefactoringDescriptor &PrimitivesInit::describe() {
  static const RefactoringDescriptor Descriptor = {
      "primitives-init",
      "Primitives Initialization",
      "Initializes a primitive variable with default value",
  };
  return Descriptor;
}

Expected<AtomicChanges>
PrimitivesInit::createSourceReplacements(RefactoringRuleContext &Context) {
  ASTContext &AST = Context.getASTContext();
  SourceManager &SM = AST.getSourceManager();
  AtomicChange Replacement(SM, Variable->getLocation());
  std::string VarName = Variable->getNameAsString();
  std::string InitWithDefaultValue = (Variable->getType()->isCharType() ?
                                                                        " = \'\\0\'" : " = 0");

  auto Error = Replacement.replace(SM, Variable->getEndLoc(),
                                   VarName.length(),
                                   VarName + InitWithDefaultValue);
  if (Error) return std::move(Error);
  return AtomicChanges{std::move(Replacement)};
}
