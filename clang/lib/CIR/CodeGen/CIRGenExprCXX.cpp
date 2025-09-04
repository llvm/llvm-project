//===--- CIRGenExprCXX.cpp - Emit CIR Code for C++ expressions ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with code generation of C++ expressions
//
//===----------------------------------------------------------------------===//

#include "CIRGenFunction.h"
#include "clang/AST/ExprCXX.h"

using namespace clang;
using namespace clang::CIRGen;

RValue CIRGenFunction::emitCXXPseudoDestructorExpr(
    const CXXPseudoDestructorExpr *expr) {
  QualType destroyedType = expr->getDestroyedType();
  if (destroyedType.hasStrongOrWeakObjCLifetime()) {
    assert(!cir::MissingFeatures::objCLifetime());
    cgm.errorNYI(expr->getExprLoc(),
                 "emitCXXPseudoDestructorExpr: Objective-C lifetime is NYI");
  } else {
    // C++ [expr.pseudo]p1:
    //   The result shall only be used as the operand for the function call
    //   operator (), and the result of such a call has type void. The only
    //   effect is the evaluation of the postfix-expression before the dot or
    //   arrow.
    emitIgnoredExpr(expr->getBase());
  }

  return RValue::get(nullptr);
}
