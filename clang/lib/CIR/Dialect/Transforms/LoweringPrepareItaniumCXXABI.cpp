//====- LoweringPrepareItaniumCXXABI.h - Itanium ABI specific code --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides Itanium C++ ABI specific code that is used during LLVMIR
// lowering prepare.
//
//===----------------------------------------------------------------------===//

#include "../IR/MissingFeatures.h"
#include "LoweringPrepareCXXABI.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

using namespace cir;

namespace {

class LoweringPrepareItaniumCXXABI : public LoweringPrepareCXXABI {
public:
  mlir::Value lowerDynamicCast(CIRBaseBuilderTy &builder,
                               mlir::cir::DynamicCastOp op) override;
};

} // namespace

LoweringPrepareCXXABI *LoweringPrepareCXXABI::createItaniumABI() {
  return new LoweringPrepareItaniumCXXABI();
}

static void buildBadCastCall(CIRBaseBuilderTy &builder, mlir::Location loc,
                             mlir::FlatSymbolRefAttr badCastFuncRef) {
  // TODO(cir): set the calling convention to __cxa_bad_cast.
  assert(!MissingFeatures::setCallingConv());

  builder.create<mlir::cir::CallOp>(loc, badCastFuncRef, mlir::ValueRange{});
  builder.create<mlir::cir::UnreachableOp>(loc);
  builder.clearInsertionPoint();
}

static mlir::Value buildDynamicCastAfterNullCheck(CIRBaseBuilderTy &builder,
                                                  mlir::cir::DynamicCastOp op) {
  auto loc = op->getLoc();
  auto srcValue = op.getSrc();
  auto castInfo = op.getInfo().cast<mlir::cir::DynamicCastInfoAttr>();

  // TODO(cir): consider address space
  assert(!MissingFeatures::addressSpace());

  auto srcPtr = builder.createBitcast(srcValue, builder.getVoidPtrTy());
  auto srcRtti = builder.getConstant(loc, castInfo.getSrcRtti());
  auto destRtti = builder.getConstant(loc, castInfo.getDestRtti());
  auto offsetHint = builder.getConstant(loc, castInfo.getOffsetHint());

  auto dynCastFuncRef = castInfo.getRuntimeFunc();
  mlir::Value dynCastFuncArgs[4] = {srcPtr, srcRtti, destRtti, offsetHint};

  // TODO(cir): set the calling convention for __dynamic_cast.
  assert(!MissingFeatures::setCallingConv());
  mlir::Value castedPtr =
      builder
          .create<mlir::cir::CallOp>(loc, dynCastFuncRef,
                                     builder.getVoidPtrTy(), dynCastFuncArgs)
          .getResult(0);

  assert(castedPtr.getType().isa<mlir::cir::PointerType>() &&
         "the return value of __dynamic_cast should be a ptr");

  /// C++ [expr.dynamic.cast]p9:
  ///   A failed cast to reference type throws std::bad_cast
  if (op.isRefcast()) {
    // Emit a cir.if that checks the casted value.
    mlir::Value castedValueIsNull = builder.createPtrIsNull(castedPtr);
    builder.create<mlir::cir::IfOp>(
        loc, castedValueIsNull, false, [&](mlir::OpBuilder &, mlir::Location) {
          buildBadCastCall(builder, loc, castInfo.getBadCastFunc());
        });
  }

  // Note that castedPtr is a void*. Cast it to a pointer to the destination
  // type before return.
  return builder.createBitcast(castedPtr, op.getType());
}

mlir::Value
LoweringPrepareItaniumCXXABI::lowerDynamicCast(CIRBaseBuilderTy &builder,
                                               mlir::cir::DynamicCastOp op) {
  auto loc = op->getLoc();
  auto srcValue = op.getSrc();

  assert(!MissingFeatures::buildTypeCheck());

  if (op.isRefcast())
    return buildDynamicCastAfterNullCheck(builder, op);

  auto srcValueIsNull = builder.createPtrToBoolCast(srcValue);
  return builder
      .create<mlir::cir::TernaryOp>(
          loc, srcValueIsNull,
          [&](mlir::OpBuilder &, mlir::Location) {
            builder.createYield(
                loc, builder.getNullPtr(op.getType(), loc).getResult());
          },
          [&](mlir::OpBuilder &, mlir::Location) {
            builder.createYield(loc,
                                buildDynamicCastAfterNullCheck(builder, op));
          })
      .getResult();
}
