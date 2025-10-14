//===--------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//
//
// This file provides Itanium C++ ABI specific code
// that is used during LLVMIR lowering prepare.
//
//===--------------------------------------------------------------------===//

#include "LoweringPrepareCXXABI.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/MissingFeatures.h"

class LoweringPrepareItaniumCXXABI : public cir::LoweringPrepareCXXABI {
public:
  mlir::Value lowerDynamicCast(cir::CIRBaseBuilderTy &builder,
                               clang::ASTContext &astCtx,
                               cir::DynamicCastOp op) override;
};

cir::LoweringPrepareCXXABI *cir::LoweringPrepareCXXABI::createItaniumABI() {
  return new LoweringPrepareItaniumCXXABI();
}

static void buildBadCastCall(cir::CIRBaseBuilderTy &builder, mlir::Location loc,
                             mlir::FlatSymbolRefAttr badCastFuncRef) {
  builder.createCallOp(loc, badCastFuncRef, cir::VoidType(),
                       mlir::ValueRange{});
  // TODO(cir): Set the 'noreturn' attribute on the function.
  assert(!cir::MissingFeatures::opFuncNoReturn());
  cir::UnreachableOp::create(builder, loc);
  builder.clearInsertionPoint();
}

static mlir::Value
buildDynamicCastAfterNullCheck(cir::CIRBaseBuilderTy &builder,
                               cir::DynamicCastOp op) {
  mlir::Location loc = op->getLoc();
  mlir::Value srcValue = op.getSrc();
  cir::DynamicCastInfoAttr castInfo = op.getInfo().value();

  // TODO(cir): consider address space
  assert(!cir::MissingFeatures::addressSpace());

  mlir::Value srcPtr = builder.createBitcast(srcValue, builder.getVoidPtrTy());
  cir::ConstantOp srcRtti = builder.getConstant(loc, castInfo.getSrcRtti());
  cir::ConstantOp destRtti = builder.getConstant(loc, castInfo.getDestRtti());
  cir::ConstantOp offsetHint =
      builder.getConstant(loc, castInfo.getOffsetHint());

  mlir::FlatSymbolRefAttr dynCastFuncRef = castInfo.getRuntimeFunc();
  mlir::Value dynCastFuncArgs[4] = {srcPtr, srcRtti, destRtti, offsetHint};

  mlir::Value castedPtr =
      builder
          .createCallOp(loc, dynCastFuncRef, builder.getVoidPtrTy(),
                        dynCastFuncArgs)
          .getResult();

  assert(mlir::isa<cir::PointerType>(castedPtr.getType()) &&
         "the return value of __dynamic_cast should be a ptr");

  /// C++ [expr.dynamic.cast]p9:
  ///   A failed cast to reference type throws std::bad_cast
  if (op.isRefCast()) {
    // Emit a cir.if that checks the casted value.
    mlir::Value castedValueIsNull = builder.createPtrIsNull(castedPtr);
    builder.create<cir::IfOp>(
        loc, castedValueIsNull, false, [&](mlir::OpBuilder &, mlir::Location) {
          buildBadCastCall(builder, loc, castInfo.getBadCastFunc());
        });
  }

  // Note that castedPtr is a void*. Cast it to a pointer to the destination
  // type before return.
  return builder.createBitcast(castedPtr, op.getType());
}

static mlir::Value
buildDynamicCastToVoidAfterNullCheck(cir::CIRBaseBuilderTy &builder,
                                     clang::ASTContext &astCtx,
                                     cir::DynamicCastOp op) {
  llvm_unreachable("dynamic cast to void is NYI");
}

mlir::Value
LoweringPrepareItaniumCXXABI::lowerDynamicCast(cir::CIRBaseBuilderTy &builder,
                                               clang::ASTContext &astCtx,
                                               cir::DynamicCastOp op) {
  mlir::Location loc = op->getLoc();
  mlir::Value srcValue = op.getSrc();

  assert(!cir::MissingFeatures::emitTypeCheck());

  if (op.isRefCast())
    return buildDynamicCastAfterNullCheck(builder, op);

  mlir::Value srcValueIsNotNull = builder.createPtrToBoolCast(srcValue);
  return builder
      .create<cir::TernaryOp>(
          loc, srcValueIsNotNull,
          [&](mlir::OpBuilder &, mlir::Location) {
            mlir::Value castedValue =
                op.isCastToVoid()
                    ? buildDynamicCastToVoidAfterNullCheck(builder, astCtx, op)
                    : buildDynamicCastAfterNullCheck(builder, op);
            builder.createYield(loc, castedValue);
          },
          [&](mlir::OpBuilder &, mlir::Location) {
            builder.createYield(
                loc, builder.getNullPtr(op.getType(), loc).getResult());
          })
      .getResult();
}
