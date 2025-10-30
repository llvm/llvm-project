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
#include "clang/CIR/MissingFeatures.h"

using namespace clang;
using namespace clang::CIRGen;

struct clang::CIRGen::CGCoroData {
  // Stores the __builtin_coro_id emitted in the function so that we can supply
  // it as the first argument to other builtins.
  cir::CallOp coroId = nullptr;

  // Stores the result of __builtin_coro_begin call.
  mlir::Value coroBegin = nullptr;
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
        cir::FuncType::get({int32Ty, voidPtrTy, voidPtrTy, voidPtrTy}, int32Ty),
        /*FD=*/nullptr);
    assert(fnOp && "should always succeed");
  } else {
    fnOp = cast<cir::FuncOp>(builtin);
  }

  return builder.createCallOp(loc, fnOp,
                              mlir::ValueRange{builder.getUInt32(newAlign, loc),
                                               nullPtr, nullPtr, nullPtr});
}

cir::CallOp CIRGenFunction::emitCoroAllocBuiltinCall(mlir::Location loc) {
  cir::BoolType boolTy = builder.getBoolTy();

  mlir::Operation *builtin = cgm.getGlobalValue(cgm.builtinCoroAlloc);

  cir::FuncOp fnOp;
  if (!builtin) {
    fnOp = cgm.createCIRBuiltinFunction(loc, cgm.builtinCoroAlloc,
                                        cir::FuncType::get({uInt32Ty}, boolTy),
                                        /*fd=*/nullptr);
    assert(fnOp && "should always succeed");
  } else {
    fnOp = cast<cir::FuncOp>(builtin);
  }

  return builder.createCallOp(
      loc, fnOp, mlir::ValueRange{curCoro.data->coroId.getResult()});
}

cir::CallOp
CIRGenFunction::emitCoroBeginBuiltinCall(mlir::Location loc,
                                         mlir::Value coroframeAddr) {
  mlir::Operation *builtin = cgm.getGlobalValue(cgm.builtinCoroBegin);

  cir::FuncOp fnOp;
  if (!builtin) {
    fnOp = cgm.createCIRBuiltinFunction(
        loc, cgm.builtinCoroBegin,
        cir::FuncType::get({uInt32Ty, voidPtrTy}, voidPtrTy),
        /*fd=*/nullptr);
    assert(fnOp && "should always succeed");
  } else {
    fnOp = cast<cir::FuncOp>(builtin);
  }

  return builder.createCallOp(
      loc, fnOp,
      mlir::ValueRange{curCoro.data->coroId.getResult(), coroframeAddr});
}

mlir::LogicalResult
CIRGenFunction::emitCoroutineBody(const CoroutineBodyStmt &s) {
  mlir::Location openCurlyLoc = getLoc(s.getBeginLoc());
  cir::ConstantOp nullPtrCst = builder.getNullPtr(voidPtrTy, openCurlyLoc);

  auto fn = mlir::cast<cir::FuncOp>(curFn);
  fn.setCoroutine(true);
  cir::CallOp coroId = emitCoroIDBuiltinCall(openCurlyLoc, nullPtrCst);
  createCoroData(*this, curCoro, coroId);

  // Backend is allowed to elide memory allocations, to help it, emit
  // auto mem = coro.alloc() ? 0 : ... allocation code ...;
  cir::CallOp coroAlloc = emitCoroAllocBuiltinCall(openCurlyLoc);

  // Initialize address of coroutine frame to null
  CanQualType astVoidPtrTy = cgm.getASTContext().VoidPtrTy;
  mlir::Type allocaTy = convertTypeForMem(astVoidPtrTy);
  Address coroFrame =
      createTempAlloca(allocaTy, getContext().getTypeAlignInChars(astVoidPtrTy),
                       openCurlyLoc, "__coro_frame_addr",
                       /*ArraySize=*/nullptr);

  mlir::Value storeAddr = coroFrame.getPointer();
  builder.CIRBaseBuilderTy::createStore(openCurlyLoc, nullPtrCst, storeAddr);
  cir::IfOp::create(
      builder, openCurlyLoc, coroAlloc.getResult(),
      /*withElseRegion=*/false,
      /*thenBuilder=*/[&](mlir::OpBuilder &b, mlir::Location loc) {
        builder.CIRBaseBuilderTy::createStore(
            loc, emitScalarExpr(s.getAllocate()), storeAddr);
        cir::YieldOp::create(builder, loc);
      });
  curCoro.data->coroBegin =
      emitCoroBeginBuiltinCall(
          openCurlyLoc,
          cir::LoadOp::create(builder, openCurlyLoc, allocaTy, storeAddr))
          .getResult();

  // Handle allocation failure if 'ReturnStmtOnAllocFailure' was provided.
  if (s.getReturnStmtOnAllocFailure())
    cgm.errorNYI("handle coroutine return alloc failure");

  assert(!cir::MissingFeatures::generateDebugInfo());
  assert(!cir::MissingFeatures::emitBodyAndFallthrough());
  return mlir::success();
}
