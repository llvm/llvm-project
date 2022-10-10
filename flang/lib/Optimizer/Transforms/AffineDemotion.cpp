//===-- AffineDemotion.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation is a prototype that demote affine dialects operations
// after optimizations to FIR loops operations.
// It is used after the AffinePromotion pass.
// It is not part of the production pipeline and would need more work in order
// to be used in production.
// More information can be found in this presentation:
// https://slides.com/rajanwalia/deck
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

namespace fir {
#define GEN_PASS_DEF_AFFINEDIALECTDEMOTION
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "flang-affine-demotion"

using namespace fir;
using namespace mlir;

namespace {

class AffineLoadConversion : public OpConversionPattern<mlir::AffineLoadOp> {
public:
  using OpConversionPattern<mlir::AffineLoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::AffineLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> indices(adaptor.getIndices());
    auto maybeExpandedMap =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!maybeExpandedMap)
      return failure();

    auto coorOp = rewriter.create<fir::CoordinateOp>(
        op.getLoc(), fir::ReferenceType::get(op.getResult().getType()),
        adaptor.getMemref(), *maybeExpandedMap);

    rewriter.replaceOpWithNewOp<fir::LoadOp>(op, coorOp.getResult());
    return success();
  }
};

class AffineStoreConversion : public OpConversionPattern<mlir::AffineStoreOp> {
public:
  using OpConversionPattern<mlir::AffineStoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::AffineStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> indices(op.getIndices());
    auto maybeExpandedMap =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!maybeExpandedMap)
      return failure();

    auto coorOp = rewriter.create<fir::CoordinateOp>(
        op.getLoc(), fir::ReferenceType::get(op.getValueToStore().getType()),
        adaptor.getMemref(), *maybeExpandedMap);
    rewriter.replaceOpWithNewOp<fir::StoreOp>(op, adaptor.getValue(),
                                              coorOp.getResult());
    return success();
  }
};

class ConvertConversion : public mlir::OpRewritePattern<fir::ConvertOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(fir::ConvertOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op.getRes().getType().isa<mlir::MemRefType>()) {
      // due to index calculation moving to affine maps we still need to
      // add converts for sequence types this has a side effect of losing
      // some information about arrays with known dimensions by creating:
      // fir.convert %arg0 : (!fir.ref<!fir.array<5xi32>>) ->
      // !fir.ref<!fir.array<?xi32>>
      if (auto refTy = op.getValue().getType().dyn_cast<fir::ReferenceType>())
        if (auto arrTy = refTy.getEleTy().dyn_cast<fir::SequenceType>()) {
          fir::SequenceType::Shape flatShape = {
              fir::SequenceType::getUnknownExtent()};
          auto flatArrTy = fir::SequenceType::get(flatShape, arrTy.getEleTy());
          auto flatTy = fir::ReferenceType::get(flatArrTy);
          rewriter.replaceOpWithNewOp<fir::ConvertOp>(op, flatTy,
                                                      op.getValue());
          return success();
        }
      rewriter.startRootUpdate(op->getParentOp());
      op.getResult().replaceAllUsesWith(op.getValue());
      rewriter.finalizeRootUpdate(op->getParentOp());
      rewriter.eraseOp(op);
    }
    return success();
  }
};

mlir::Type convertMemRef(mlir::MemRefType type) {
  return fir::SequenceType::get(
      SmallVector<int64_t>(type.getShape().begin(), type.getShape().end()),
      type.getElementType());
}

class StdAllocConversion : public mlir::OpRewritePattern<memref::AllocOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(memref::AllocOp op,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<fir::AllocaOp>(op, convertMemRef(op.getType()),
                                               op.getMemref());
    return success();
  }
};

class AffineDialectDemotion
    : public fir::impl::AffineDialectDemotionBase<AffineDialectDemotion> {
public:
  void runOnOperation() override {
    auto *context = &getContext();
    auto function = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "AffineDemotion: running on function:\n";
               function.print(llvm::dbgs()););

    mlir::RewritePatternSet patterns(context);
    patterns.insert<ConvertConversion>(context);
    patterns.insert<AffineLoadConversion>(context);
    patterns.insert<AffineStoreConversion>(context);
    patterns.insert<StdAllocConversion>(context);
    mlir::ConversionTarget target(*context);
    target.addIllegalOp<memref::AllocOp>();
    target.addDynamicallyLegalOp<fir::ConvertOp>([](fir::ConvertOp op) {
      if (op.getRes().getType().isa<mlir::MemRefType>())
        return false;
      return true;
    });
    target
        .addLegalDialect<FIROpsDialect, mlir::scf::SCFDialect,
                         mlir::arith::ArithDialect, mlir::func::FuncDialect>();

    if (mlir::failed(mlir::applyPartialConversion(function, target,
                                                  std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "error in converting affine dialect\n");
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> fir::createAffineDemotionPass() {
  return std::make_unique<AffineDialectDemotion>();
}
