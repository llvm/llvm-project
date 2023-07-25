//===- SimplifyHLFIRIntrinsics.cpp - Simplify HLFIR Intrinsics ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Normally transformational intrinsics are lowered to calls to runtime
// functions. However, some cases of the intrinsics are faster when inlined
// into the calling function.
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace hlfir {
#define GEN_PASS_DEF_SIMPLIFYHLFIRINTRINSICS
#include "flang/Optimizer/HLFIR/Passes.h.inc"
} // namespace hlfir

namespace {

class TransposeAsElementalConversion
    : public mlir::OpRewritePattern<hlfir::TransposeOp> {
public:
  using mlir::OpRewritePattern<hlfir::TransposeOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hlfir::TransposeOp transpose,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = transpose.getLoc();
    fir::FirOpBuilder builder{rewriter, transpose.getOperation()};
    hlfir::ExprType expr = transpose.getType();
    mlir::Type elementType = expr.getElementType();
    hlfir::Entity array = hlfir::Entity{transpose.getArray()};
    mlir::Value resultShape = genResultShape(loc, builder, array);
    llvm::SmallVector<mlir::Value, 1> typeParams;
    hlfir::genLengthParameters(loc, builder, array, typeParams);

    auto genKernel = [&array](mlir::Location loc, fir::FirOpBuilder &builder,
                              mlir::ValueRange inputIndices) -> hlfir::Entity {
      assert(inputIndices.size() == 2 && "checked in TransposeOp::validate");
      mlir::ValueRange transposedIndices{{inputIndices[1], inputIndices[0]}};
      hlfir::Entity element =
          hlfir::getElementAt(loc, builder, array, transposedIndices);
      hlfir::Entity val = hlfir::loadTrivialScalar(loc, builder, element);
      return val;
    };
    hlfir::ElementalOp elementalOp = hlfir::genElementalOp(
        loc, builder, elementType, resultShape, typeParams, genKernel,
        /*isUnordered=*/true, transpose.getResult().getType());

    // it wouldn't be safe to replace block arguments with a different
    // hlfir.expr type. Types can differ due to differing amounts of shape
    // information
    assert(elementalOp.getResult().getType() ==
           transpose.getResult().getType());

    rewriter.replaceOp(transpose, elementalOp);
    return mlir::success();
  }

private:
  static mlir::Value genResultShape(mlir::Location loc,
                                    fir::FirOpBuilder &builder,
                                    hlfir::Entity array) {
    mlir::Value inShape = hlfir::genShape(loc, builder, array);
    llvm::SmallVector<mlir::Value> inExtents =
        hlfir::getExplicitExtentsFromShape(inShape, builder);
    if (inShape.getUses().empty())
      inShape.getDefiningOp()->erase();

    // transpose indices
    assert(inExtents.size() == 2 && "checked in TransposeOp::validate");
    return builder.create<fir::ShapeOp>(
        loc, mlir::ValueRange{inExtents[1], inExtents[0]});
  }
};

class SimplifyHLFIRIntrinsics
    : public hlfir::impl::SimplifyHLFIRIntrinsicsBase<SimplifyHLFIRIntrinsics> {
public:
  void runOnOperation() override {
    mlir::func::FuncOp func = this->getOperation();
    mlir::MLIRContext *context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.insert<TransposeAsElementalConversion>(context);
    mlir::ConversionTarget target(*context);
    // don't transform transpose of polymorphic arrays (not currently supported
    // by hlfir.elemental)
    target.addDynamicallyLegalOp<hlfir::TransposeOp>(
        [](hlfir::TransposeOp transpose) {
          return transpose.getType().cast<hlfir::ExprType>().isPolymorphic();
        });
    target.markUnknownOpDynamicallyLegal(
        [](mlir::Operation *) { return true; });
    if (mlir::failed(
            mlir::applyFullConversion(func, target, std::move(patterns)))) {
      mlir::emitError(func->getLoc(),
                      "failure in HLFIR intrinsic simplification");
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> hlfir::createSimplifyHLFIRIntrinsicsPass() {
  return std::make_unique<SimplifyHLFIRIntrinsics>();
}
