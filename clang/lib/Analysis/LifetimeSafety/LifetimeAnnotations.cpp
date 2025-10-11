//===- LifetimeAnnotations.cpp -  -*--------------- C++------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "clang/Analysis/Analyses/LifetimeSafety/LifetimeAnnotations.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"

namespace clang::lifetimes {

const FunctionDecl *
getDeclWithMergedLifetimeBoundAttrs(const FunctionDecl *FD) {
  return FD != nullptr ? FD->getMostRecentDecl() : nullptr;
}

const CXXMethodDecl *
getDeclWithMergedLifetimeBoundAttrs(const CXXMethodDecl *CMD) {
  const FunctionDecl *FD = CMD;
  return cast_if_present<CXXMethodDecl>(
      getDeclWithMergedLifetimeBoundAttrs(FD));
}

bool isNormalAssignmentOperator(const FunctionDecl *FD) {
  OverloadedOperatorKind OO = FD->getDeclName().getCXXOverloadedOperator();
  bool IsAssignment = OO == OO_Equal || isCompoundAssignmentOperator(OO);
  if (!IsAssignment)
    return false;
  QualType RetT = FD->getReturnType();
  if (!RetT->isLValueReferenceType())
    return false;
  ASTContext &Ctx = FD->getASTContext();
  QualType LHST;
  auto *MD = dyn_cast<CXXMethodDecl>(FD);
  if (MD && MD->isCXXInstanceMember())
    LHST = Ctx.getLValueReferenceType(MD->getFunctionObjectParameterType());
  else
    LHST = FD->getParamDecl(0)->getType();
  return Ctx.hasSameType(RetT, LHST);
}

bool isAssignmentOperatorLifetimeBound(const CXXMethodDecl *CMD) {
  CMD = getDeclWithMergedLifetimeBoundAttrs(CMD);
  return CMD && isNormalAssignmentOperator(CMD) && CMD->param_size() == 1 &&
         CMD->getParamDecl(0)->hasAttr<clang::LifetimeBoundAttr>();
}

bool implicitObjectParamIsLifetimeBound(const FunctionDecl *FD) {
  FD = getDeclWithMergedLifetimeBoundAttrs(FD);
  const TypeSourceInfo *TSI = FD->getTypeSourceInfo();
  if (!TSI)
    return false;
  // Don't declare this variable in the second operand of the for-statement;
  // GCC miscompiles that by ending its lifetime before evaluating the
  // third operand. See gcc.gnu.org/PR86769.
  AttributedTypeLoc ATL;
  for (TypeLoc TL = TSI->getTypeLoc();
       (ATL = TL.getAsAdjusted<AttributedTypeLoc>());
       TL = ATL.getModifiedLoc()) {
    if (ATL.getAttrAs<clang::LifetimeBoundAttr>())
      return true;
  }

  return isNormalAssignmentOperator(FD);
}

} // namespace clang::lifetimes
