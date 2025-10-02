//===----- CGCoroutine.cpp - Emit CIR Code for C++ coroutines -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ code generation of coroutines.
//
//===----------------------------------------------------------------------===//

#include "CIRGenFunction.h"
#include "mlir/Support/LLVM.h"
#include "clang/AST/StmtCXX.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

using namespace clang;
using namespace clang::CIRGen;

struct clang::CIRGen::CGCoroData {
  // Stores the __builtin_coro_id emitted in the function so that we can supply
  // it as the first argument to other builtins.
  cir::CallOp coroId = nullptr;
};

// Defining these here allows to keep CGCoroData private to this file.
CIRGenFunction::CGCoroInfo::CGCoroInfo() {}
CIRGenFunction::CGCoroInfo::~CGCoroInfo() {}

static void createCoroData(CIRGenFunction &cgf,
                           CIRGenFunction::CGCoroInfo &curCoro,
                           cir::CallOp coroId) {
  assert(!curCoro.data && "EmitCoroutineBodyStatement called twice?");

  curCoro.data = std::make_unique<CGCoroData>();
  curCoro.data->coroId = coroId;
}

cir::CallOp CIRGenFunction::emitCoroIDBuiltinCall(mlir::Location loc,
                                                  mlir::Value nullPtr) {
  cir::IntType int32Ty = builder.getUInt32Ty();

  const TargetInfo &ti = cgm.getASTContext().getTargetInfo();
  unsigned newAlign = ti.getNewAlign() / ti.getCharWidth();

  mlir::Operation *builtin = cgm.getGlobalValue(cgm.builtinCoroId);

  cir::FuncOp fnOp;
  if (!builtin) {
    fnOp = cgm.createCIRBuiltinFunction(
        loc, cgm.builtinCoroId,
        cir::FuncType::get({int32Ty, VoidPtrTy, VoidPtrTy, VoidPtrTy}, int32Ty),
        /*FD=*/nullptr);
    assert(fnOp && "should always succeed");
  } else {
    fnOp = cast<cir::FuncOp>(builtin);
  }

  return builder.createCallOp(loc, fnOp,
                              mlir::ValueRange{builder.getUInt32(newAlign, loc),
                                               nullPtr, nullPtr, nullPtr});
}

mlir::LogicalResult
CIRGenFunction::emitCoroutineBody(const CoroutineBodyStmt &s) {
  mlir::Location openCurlyLoc = getLoc(s.getBeginLoc());
  cir::ConstantOp nullPtrCst = builder.getNullPtr(VoidPtrTy, openCurlyLoc);

  auto fn = mlir::cast<cir::FuncOp>(curFn);
  fn.setCoroutine(true);
  cir::CallOp coroId = emitCoroIDBuiltinCall(openCurlyLoc, nullPtrCst);
  createCoroData(*this, curCoro, coroId);

  assert(!cir::MissingFeatures::coroAllocBuiltinCall());

  assert(!cir::MissingFeatures::coroBeginBuiltinCall());

  assert(!cir::MissingFeatures::generateDebugInfo());
  return mlir::success();
}
