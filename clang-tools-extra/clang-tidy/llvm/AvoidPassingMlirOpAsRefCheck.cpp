//===--- AvoidPassingMlirOpAsRefCheck.cpp - clang-tidy --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidPassingMlirOpAsRefCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::llvm_check {

void AvoidPassingMlirOpAsRefCheck::registerMatchers(MatchFinder *Finder) {
  // Match parameters that are references to classes derived from mlir::Op.
  Finder->addMatcher(
      parmVarDecl(
          hasType(qualType(
              references(qualType(hasDeclaration(
                  cxxRecordDecl(isSameOrDerivedFrom("::mlir::Op"))
                      .bind("op_type")))),
              // We want to avoid matching `Op *&` (reference to pointer to Op)
              // which is not common for Op but possible.
              unless(references(pointerType())))),
          unless(isImplicit()),
          unless(hasAncestor(cxxConstructorDecl(
              anyOf(isCopyConstructor(), isMoveConstructor())))),
          unless(hasAncestor(cxxMethodDecl(isCopyAssignmentOperator()))),
          unless(hasAncestor(cxxMethodDecl(isMoveAssignmentOperator()))),
          decl().bind("param")),
      this);
}

void AvoidPassingMlirOpAsRefCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Param = Result.Nodes.getNodeAs<ParmVarDecl>("param");
  const auto *OpType = Result.Nodes.getNodeAs<CXXRecordDecl>("op_type");

  // Exclude if we can't find definitions.
  if (!Param || !OpType)
    return;

  diag(Param->getLocation(),
       "MLIR Op class '%0' should be passed by value, not by reference")
      << OpType->getName();
}

} // namespace clang::tidy::llvm_check
