//===- LoweringPrepare.cpp - pareparation work for LLVM lowering ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CharUnits.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/MissingFeatures.h"

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
  void lowerArrayDtor(cir::ArrayDtor op);
  void lowerArrayCtor(cir::ArrayCtor op);

  ///
  /// AST related
  /// -----------

  clang::ASTContext *astCtx;

  void setASTContext(clang::ASTContext *c) { astCtx = c; }
};

} // namespace

static mlir::Value lowerScalarToComplexCast(mlir::MLIRContext &ctx,
                                            cir::CastOp op) {
  cir::CIRBaseBuilderTy builder(ctx);
  builder.setInsertionPoint(op);

  mlir::Value src = op.getSrc();
  mlir::Value imag = builder.getNullValue(src.getType(), op.getLoc());
  return builder.createComplexCreate(op.getLoc(), src, imag);
}

static mlir::Value lowerComplexToScalarCast(mlir::MLIRContext &ctx,
                                            cir::CastOp op,
                                            cir::CastKind elemToBoolKind) {
  cir::CIRBaseBuilderTy builder(ctx);
  builder.setInsertionPoint(op);

  mlir::Value src = op.getSrc();
  if (!mlir::isa<cir::BoolType>(op.getType()))
    return builder.createComplexReal(op.getLoc(), src);

  // Complex cast to bool: (bool)(a+bi) => (bool)a || (bool)b
  mlir::Value srcReal = builder.createComplexReal(op.getLoc(), src);
  mlir::Value srcImag = builder.createComplexImag(op.getLoc(), src);

  cir::BoolType boolTy = builder.getBoolTy();
  mlir::Value srcRealToBool =
      builder.createCast(op.getLoc(), elemToBoolKind, srcReal, boolTy);
  mlir::Value srcImagToBool =
      builder.createCast(op.getLoc(), elemToBoolKind, srcImag, boolTy);
  return builder.createLogicalOr(op.getLoc(), srcRealToBool, srcImagToBool);
}

static mlir::Value lowerComplexToComplexCast(mlir::MLIRContext &ctx,
                                             cir::CastOp op,
                                             cir::CastKind scalarCastKind) {
  CIRBaseBuilderTy builder(ctx);
  builder.setInsertionPoint(op);

  mlir::Value src = op.getSrc();
  auto dstComplexElemTy =
      mlir::cast<cir::ComplexType>(op.getType()).getElementType();

  mlir::Value srcReal = builder.createComplexReal(op.getLoc(), src);
  mlir::Value srcImag = builder.createComplexImag(op.getLoc(), src);

  mlir::Value dstReal = builder.createCast(op.getLoc(), scalarCastKind, srcReal,
                                           dstComplexElemTy);
  mlir::Value dstImag = builder.createCast(op.getLoc(), scalarCastKind, srcImag,
                                           dstComplexElemTy);
  return builder.createComplexCreate(op.getLoc(), dstReal, dstImag);
}

void LoweringPreparePass::lowerCastOp(cir::CastOp op) {
  mlir::MLIRContext &ctx = getContext();
  mlir::Value loweredValue = [&]() -> mlir::Value {
    switch (op.getKind()) {
    case cir::CastKind::float_to_complex:
    case cir::CastKind::int_to_complex:
      return lowerScalarToComplexCast(ctx, op);
    case cir::CastKind::float_complex_to_real:
    case cir::CastKind::int_complex_to_real:
      return lowerComplexToScalarCast(ctx, op, op.getKind());
    case cir::CastKind::float_complex_to_bool:
      return lowerComplexToScalarCast(ctx, op, cir::CastKind::float_to_bool);
    case cir::CastKind::int_complex_to_bool:
      return lowerComplexToScalarCast(ctx, op, cir::CastKind::int_to_bool);
    case cir::CastKind::float_complex:
      return lowerComplexToComplexCast(ctx, op, cir::CastKind::floating);
    case cir::CastKind::float_complex_to_int_complex:
      return lowerComplexToComplexCast(ctx, op, cir::CastKind::float_to_int);
    case cir::CastKind::int_complex:
      return lowerComplexToComplexCast(ctx, op, cir::CastKind::integral);
    case cir::CastKind::int_complex_to_float_complex:
      return lowerComplexToComplexCast(ctx, op, cir::CastKind::int_to_float);
    default:
      return nullptr;
    }
  }();

  if (loweredValue) {
    op.replaceAllUsesWith(loweredValue);
    op.erase();
  }
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

static void lowerArrayDtorCtorIntoLoop(cir::CIRBaseBuilderTy &builder,
                                       clang::ASTContext *astCtx,
                                       mlir::Operation *op, mlir::Type eltTy,
                                       mlir::Value arrayAddr, uint64_t arrayLen,
                                       bool isCtor) {
  // Generate loop to call into ctor/dtor for every element.
  mlir::Location loc = op->getLoc();

  // TODO: instead of getting the size from the AST context, create alias for
  // PtrDiffTy and unify with CIRGen stuff.
  const unsigned sizeTypeSize =
      astCtx->getTypeSize(astCtx->getSignedSizeType());
  uint64_t endOffset = isCtor ? arrayLen : arrayLen - 1;
  mlir::Value endOffsetVal =
      builder.getUnsignedInt(loc, endOffset, sizeTypeSize);

  auto begin = cir::CastOp::create(builder, loc, eltTy,
                                   cir::CastKind::array_to_ptrdecay, arrayAddr);
  mlir::Value end =
      cir::PtrStrideOp::create(builder, loc, eltTy, begin, endOffsetVal);
  mlir::Value start = isCtor ? begin : end;
  mlir::Value stop = isCtor ? end : begin;

  mlir::Value tmpAddr = builder.createAlloca(
      loc, /*addr type*/ builder.getPointerTo(eltTy),
      /*var type*/ eltTy, "__array_idx", builder.getAlignmentAttr(1));
  builder.createStore(loc, start, tmpAddr);

  cir::DoWhileOp loop = builder.createDoWhile(
      loc,
      /*condBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        auto currentElement = b.create<cir::LoadOp>(loc, eltTy, tmpAddr);
        mlir::Type boolTy = cir::BoolType::get(b.getContext());
        auto cmp = builder.create<cir::CmpOp>(loc, boolTy, cir::CmpOpKind::ne,
                                              currentElement, stop);
        builder.createCondition(cmp);
      },
      /*bodyBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        auto currentElement = b.create<cir::LoadOp>(loc, eltTy, tmpAddr);

        cir::CallOp ctorCall;
        op->walk([&](cir::CallOp c) { ctorCall = c; });
        assert(ctorCall && "expected ctor call");

        // Array elements get constructed in order but destructed in reverse.
        mlir::Value stride;
        if (isCtor)
          stride = builder.getUnsignedInt(loc, 1, sizeTypeSize);
        else
          stride = builder.getSignedInt(loc, -1, sizeTypeSize);

        ctorCall->moveBefore(stride.getDefiningOp());
        ctorCall->setOperand(0, currentElement);
        auto nextElement = cir::PtrStrideOp::create(builder, loc, eltTy,
                                                    currentElement, stride);

        // Store the element pointer to the temporary variable
        builder.createStore(loc, nextElement, tmpAddr);
        builder.createYield(loc);
      });

  op->replaceAllUsesWith(loop);
  op->erase();
}

void LoweringPreparePass::lowerArrayDtor(cir::ArrayDtor op) {
  CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointAfter(op.getOperation());

  mlir::Type eltTy = op->getRegion(0).getArgument(0).getType();
  assert(!cir::MissingFeatures::vlas());
  auto arrayLen =
      mlir::cast<cir::ArrayType>(op.getAddr().getType().getPointee()).getSize();
  lowerArrayDtorCtorIntoLoop(builder, astCtx, op, eltTy, op.getAddr(), arrayLen,
                             false);
}

void LoweringPreparePass::lowerArrayCtor(cir::ArrayCtor op) {
  cir::CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointAfter(op.getOperation());

  mlir::Type eltTy = op->getRegion(0).getArgument(0).getType();
  assert(!cir::MissingFeatures::vlas());
  auto arrayLen =
      mlir::cast<cir::ArrayType>(op.getAddr().getType().getPointee()).getSize();
  lowerArrayDtorCtorIntoLoop(builder, astCtx, op, eltTy, op.getAddr(), arrayLen,
                             true);
}

void LoweringPreparePass::runOnOp(mlir::Operation *op) {
  if (auto arrayCtor = dyn_cast<ArrayCtor>(op))
    lowerArrayCtor(arrayCtor);
  else if (auto arrayDtor = dyn_cast<cir::ArrayDtor>(op))
    lowerArrayDtor(arrayDtor);
  else if (auto cast = mlir::dyn_cast<cir::CastOp>(op))
    lowerCastOp(cast);
  else if (auto unary = mlir::dyn_cast<cir::UnaryOp>(op))
    lowerUnaryOp(unary);
}

void LoweringPreparePass::runOnOperation() {
  mlir::Operation *op = getOperation();

  llvm::SmallVector<mlir::Operation *> opsToTransform;

  op->walk([&](mlir::Operation *op) {
    if (mlir::isa<cir::ArrayCtor, cir::ArrayDtor, cir::CastOp, cir::UnaryOp>(
            op))
      opsToTransform.push_back(op);
  });

  for (mlir::Operation *o : opsToTransform)
    runOnOp(o);
}

std::unique_ptr<Pass> mlir::createLoweringPreparePass() {
  return std::make_unique<LoweringPreparePass>();
}

std::unique_ptr<Pass>
mlir::createLoweringPreparePass(clang::ASTContext *astCtx) {
  auto pass = std::make_unique<LoweringPreparePass>();
  pass->setASTContext(astCtx);
  return std::move(pass);
}
