//===- LoweringPrepare.cpp - pareparation work for LLVM lowering ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "clang/AST/ASTContext.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/Passes.h"

#include <iostream>
#include <memory>

using namespace mlir;
using namespace cir;

namespace {
struct LoweringPreparePass : public LoweringPrepareBase<LoweringPreparePass> {
  LoweringPreparePass() = default;
  void runOnOperation() override;

  void runOnOp(mlir::Operation *op);
  void lowerCastOp(cir::CastOp op);
  void lowerUnaryOp(cir::UnaryOp op);
};

} // namespace

static mlir::Value lowerScalarToComplexCast(MLIRContext &ctx, CastOp op) {
  CIRBaseBuilderTy builder(ctx);
  builder.setInsertionPoint(op);

  mlir::Value src = op.getSrc();
  mlir::Value imag = builder.getNullValue(src.getType(), op.getLoc());
  return builder.createComplexCreate(op.getLoc(), src, imag);
}

static mlir::Value lowerComplexToScalarCast(MLIRContext &ctx, CastOp op) {
  CIRBaseBuilderTy builder(ctx);
  builder.setInsertionPoint(op);

  mlir::Value src = op.getSrc();
  if (!mlir::isa<cir::BoolType>(op.getType()))
    return builder.createComplexReal(op.getLoc(), src);

  // Complex cast to bool: (bool)(a+bi) => (bool)a || (bool)b
  mlir::Value srcReal = builder.createComplexReal(op.getLoc(), src);
  mlir::Value srcImag = builder.createComplexImag(op.getLoc(), src);

  cir::CastKind elemToBoolKind;
  if (op.getKind() == cir::CastKind::float_complex_to_bool)
    elemToBoolKind = cir::CastKind::float_to_bool;
  else if (op.getKind() == cir::CastKind::int_complex_to_bool)
    elemToBoolKind = cir::CastKind::int_to_bool;
  else
    llvm_unreachable("invalid complex to bool cast kind");

  cir::BoolType boolTy = builder.getBoolTy();
  mlir::Value srcRealToBool =
      builder.createCast(op.getLoc(), elemToBoolKind, srcReal, boolTy);
  mlir::Value srcImagToBool =
      builder.createCast(op.getLoc(), elemToBoolKind, srcImag, boolTy);

  return builder.createLogicalOr(op.getLoc(), srcRealToBool, srcImagToBool);
}

static mlir::Value lowerComplexToComplexCast(MLIRContext &ctx, CastOp op) {
  CIRBaseBuilderTy builder(ctx);
  builder.setInsertionPoint(op);

  mlir::Value src = op.getSrc();
  auto dstComplexElemTy =
      mlir::cast<cir::ComplexType>(op.getType()).getElementType();

  mlir::Value srcReal = builder.createComplexReal(op.getLoc(), src);
  mlir::Value srcImag = builder.createComplexReal(op.getLoc(), src);

  cir::CastKind scalarCastKind;
  switch (op.getKind()) {
  case cir::CastKind::float_complex:
    scalarCastKind = cir::CastKind::floating;
    break;
  case cir::CastKind::float_complex_to_int_complex:
    scalarCastKind = cir::CastKind::float_to_int;
    break;
  case cir::CastKind::int_complex:
    scalarCastKind = cir::CastKind::integral;
    break;
  case cir::CastKind::int_complex_to_float_complex:
    scalarCastKind = cir::CastKind::int_to_float;
    break;
  default:
    llvm_unreachable("invalid complex to complex cast kind");
  }

  mlir::Value dstReal = builder.createCast(op.getLoc(), scalarCastKind, srcReal,
                                           dstComplexElemTy);
  mlir::Value dstImag = builder.createCast(op.getLoc(), scalarCastKind, srcImag,
                                           dstComplexElemTy);
  return builder.createComplexCreate(op.getLoc(), dstReal, dstImag);
}

void LoweringPreparePass::lowerCastOp(cir::CastOp op) {
  mlir::Value loweredValue;
  switch (op.getKind()) {
  case cir::CastKind::float_to_complex:
  case cir::CastKind::int_to_complex:
    loweredValue = lowerScalarToComplexCast(getContext(), op);
    break;

  case cir::CastKind::float_complex_to_real:
  case cir::CastKind::int_complex_to_real:
  case cir::CastKind::float_complex_to_bool:
  case cir::CastKind::int_complex_to_bool: {
    loweredValue = lowerComplexToScalarCast(getContext(), op);
    break;
  }

  case cir::CastKind::float_complex:
  case cir::CastKind::float_complex_to_int_complex:
  case cir::CastKind::int_complex:
  case cir::CastKind::int_complex_to_float_complex:
    loweredValue = lowerComplexToComplexCast(getContext(), op);
    break;

  default:
    return;
  }

  op.replaceAllUsesWith(loweredValue);
  op.erase();
}

void LoweringPreparePass::lowerUnaryOp(cir::UnaryOp op) {
  mlir::Type ty = op.getType();
  if (!mlir::isa<cir::ComplexType>(ty))
    return;

  mlir::Location loc = op.getLoc();
  cir::UnaryOpKind opKind = op.getKind();

  CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointAfter(op);

  mlir::Value operand = op.getInput();
  mlir::Value operandReal = builder.createComplexReal(loc, operand);
  mlir::Value operandImag = builder.createComplexImag(loc, operand);

  mlir::Value resultReal;
  mlir::Value resultImag;

  switch (opKind) {
  case cir::UnaryOpKind::Inc:
  case cir::UnaryOpKind::Dec:
    resultReal = builder.createUnaryOp(loc, opKind, operandReal);
    resultImag = operandImag;
    break;

  case cir::UnaryOpKind::Plus:
  case cir::UnaryOpKind::Minus:
    llvm_unreachable("Complex unary Plus/Minus NYI");
    break;

  case cir::UnaryOpKind::Not:
    resultReal = operandReal;
    resultImag =
        builder.createUnaryOp(loc, cir::UnaryOpKind::Minus, operandImag);
    break;
  }

  mlir::Value result = builder.createComplexCreate(loc, resultReal, resultImag);
  op.replaceAllUsesWith(result);
  op.erase();
}

void LoweringPreparePass::runOnOp(mlir::Operation *op) {
  if (auto cast = dyn_cast<cir::CastOp>(op)) {
    lowerCastOp(cast);
  } else if (auto unary = dyn_cast<cir::UnaryOp>(op)) {
    lowerUnaryOp(unary);
  }
}

void LoweringPreparePass::runOnOperation() {
  mlir::Operation *op = getOperation();

  llvm::SmallVector<mlir::Operation *> opsToTransform;

  op->walk([&](mlir::Operation *op) {
    if (mlir::isa<cir::CastOp, cir::UnaryOp>(op))
      opsToTransform.push_back(op);
  });

  for (mlir::Operation *o : opsToTransform)
    runOnOp(o);
}

std::unique_ptr<Pass> mlir::createLoweringPreparePass() {
  return std::make_unique<LoweringPreparePass>();
}
