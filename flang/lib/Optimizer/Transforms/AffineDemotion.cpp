//===-- AffineDemotion.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "flang-affine-demotion"

using namespace fir;

namespace {

class AffineLoadConversion : public OpRewritePattern<mlir::AffineLoadOp> {
public:
  using OpRewritePattern<mlir::AffineLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::AffineLoadOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> indices(op.getMapOperands());
    auto maybeExpandedMap =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!maybeExpandedMap)
      return failure();

    auto coorOp = rewriter.create<fir::CoordinateOp>(
        op.getLoc(), fir::ReferenceType::get(op.getResult().getType()),
        op.getMemRef(), *maybeExpandedMap);

    rewriter.replaceOpWithNewOp<fir::LoadOp>(op, coorOp.getResult());
    return success();
  }
};

class AffineStoreConversion : public OpRewritePattern<mlir::AffineStoreOp> {
public:
  using OpRewritePattern<mlir::AffineStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::AffineStoreOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> indices(op.getMapOperands());
    auto maybeExpandedMap =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!maybeExpandedMap)
      return failure();

    auto coorOp = rewriter.create<fir::CoordinateOp>(
        op.getLoc(), fir::ReferenceType::get(op.getValueToStore().getType()),
        op.getMemRef(), *maybeExpandedMap);
    rewriter.replaceOpWithNewOp<fir::StoreOp>(op, op.getValueToStore(),
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
    if (op.res().getType().isa<mlir::MemRefType>()) {
      // due to index calculation moving to affine maps we still need to
      // add converts for sequence types this has a side effect of losing
      // some information about arrays with known dimensions by creating:
      // fir.convert %arg0 : (!fir.ref<!fir.array<5xi32>>) ->
      // !fir.ref<!fir.array<?xi32>>
      if (auto refTy = op.value().getType().dyn_cast<fir::ReferenceType>())
        if (auto arrTy = refTy.getEleTy().dyn_cast<fir::SequenceType>()) {
          fir::SequenceType::Shape flatShape = {
              fir::SequenceType::getUnknownExtent()};
          auto flatArrTy = fir::SequenceType::get(flatShape, arrTy.getEleTy());
          auto flatTy = fir::ReferenceType::get(flatArrTy);
          rewriter.replaceOpWithNewOp<fir::ConvertOp>(op, flatTy, op.value());
          return success();
        }
      rewriter.startRootUpdate(op->getParentOp());
      op.getResult().replaceAllUsesWith(op.value());
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
                                               op.memref());
    return success();
  }
};

class AffineDialectDemotion
    : public AffineDialectDemotionBase<AffineDialectDemotion> {
public:
  void runOnFunction() override {
    auto *context = &getContext();
    auto function = getFunction();
    LLVM_DEBUG(llvm::dbgs() << "AffineDemotion: running on function:\n";
               function.print(llvm::dbgs()););

    mlir::OwningRewritePatternList patterns(context);
    patterns.insert<ConvertConversion>(context);
    patterns.insert<AffineLoadConversion>(context);
    patterns.insert<AffineStoreConversion>(context);
    patterns.insert<StdAllocConversion>(context);
    mlir::ConversionTarget target(*context);
    target.addIllegalOp<memref::AllocOp>();
    target.addDynamicallyLegalOp<fir::ConvertOp>([](fir::ConvertOp op) {
      if (op.res().getType().isa<mlir::MemRefType>())
        return false;
      return true;
    });
    target.addLegalDialect<FIROpsDialect, mlir::scf::SCFDialect,
                           mlir::StandardOpsDialect>();

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
