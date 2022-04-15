//===--- CIRGenExprAgg.cpp - Emit CIR Code from Aggregate Expressions -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Aggregate Expr nodes as CIR code.
//
//===----------------------------------------------------------------------===//

#include "CIRGenFunction.h"
using namespace cir;
using namespace clang;
void CIRGenFunction::buildAggExpr(const Expr *E, AggValueSlot Slot) {
  assert(E && CIRGenFunction::hasAggregateEvaluationKind(E->getType()) &&
         "Invalid aggregate expression to emit");
  assert((Slot.getAddress().isValid() || Slot.isIgnored()) &&
         "slot has bits but no address");

  // TODO: assert(false && "Figure out how to assert we're in c++");
  if (const RecordType *RT = CGM.getASTContext()
                                 .getBaseElementType(E->getType())
                                 ->getAs<RecordType>()) {
    auto *RD = cast<CXXRecordDecl>(RT->getDecl());
    assert(RD->hasUserDeclaredConstructor() &&
           "default constructors aren't expected here YET");
  }

  llvm_unreachable("NYI");
}
