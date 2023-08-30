//===--- CIRGenException.cpp - Emit CIR Code for C++ exceptions -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ exception related code generation.
//
//===----------------------------------------------------------------------===//

#include "CIRDataLayout.h"
#include "CIRGenCXXABI.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "UnimplementedFeatureGuarding.h"

#include "clang/AST/StmtVisitor.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

using namespace cir;
using namespace clang;

void CIRGenFunction::buildCXXThrowExpr(const CXXThrowExpr *E) {
  if (const Expr *SubExpr = E->getSubExpr()) {
    QualType ThrowType = SubExpr->getType();
    if (ThrowType->isObjCObjectPointerType()) {
      llvm_unreachable("NYI");
    } else {
      CGM.getCXXABI().buildThrow(*this, E);
    }
  } else {
    CGM.getCXXABI().buildRethrow(*this, /*isNoReturn=*/true);
  }

  // In LLVM codegen the expression emitters expect to leave this
  // path by starting a new basic block. We do not need that in CIR.
}

namespace {
/// A cleanup to free the exception object if its initialization
/// throws.
struct FreeException final : EHScopeStack::Cleanup {
  mlir::Value exn;
  FreeException(mlir::Value exn) : exn(exn) {}
  void Emit(CIRGenFunction &CGF, Flags flags) override {
    llvm_unreachable("call to cxa_free or equivalent op NYI");
  }
};
} // end anonymous namespace

// Emits an exception expression into the given location.  This
// differs from buildAnyExprToMem only in that, if a final copy-ctor
// call is required, an exception within that copy ctor causes
// std::terminate to be invoked.
void CIRGenFunction::buildAnyExprToExn(const Expr *e, Address addr) {
  // Make sure the exception object is cleaned up if there's an
  // exception during initialization.
  pushFullExprCleanup<FreeException>(EHCleanup, addr.getPointer());
  EHScopeStack::stable_iterator cleanup = EHStack.stable_begin();

  // __cxa_allocate_exception returns a void*;  we need to cast this
  // to the appropriate type for the object.
  auto ty = convertTypeForMem(e->getType());
  Address typedAddr = addr.withElementType(ty);

  // From LLVM's codegen:
  // FIXME: this isn't quite right!  If there's a final unelided call
  // to a copy constructor, then according to [except.terminate]p1 we
  // must call std::terminate() if that constructor throws, because
  // technically that copy occurs after the exception expression is
  // evaluated but before the exception is caught.  But the best way
  // to handle that is to teach EmitAggExpr to do the final copy
  // differently if it can't be elided.
  buildAnyExprToMem(e, typedAddr, e->getType().getQualifiers(),
                    /*IsInit*/ true);

  // Deactivate the cleanup block.
  auto op = typedAddr.getPointer().getDefiningOp();
  assert(op &&
         "expected valid Operation *, block arguments are not meaningful here");
  DeactivateCleanupBlock(cleanup, op);
}