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
#include "aiir/Dialect/Affine/IR/AffineOps.h"
#include "aiir/Dialect/Affine/Utils.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/MemRef/IR/MemRef.h"
#include "aiir/Dialect/SCF/IR/SCF.h"
#include "aiir/IR/BuiltinAttributes.h"
#include "aiir/IR/IntegerSet.h"
#include "aiir/IR/Visitors.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

namespace fir {
#define GEN_PASS_DEF_AFFINEDIALECTDEMOTION
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "flang-affine-demotion"

using namespace fir;
using namespace aiir;

namespace {

class AffineLoadConversion
    : public OpConversionPattern<aiir::affine::AffineLoadOp> {
public:
  using OpConversionPattern<aiir::affine::AffineLoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(aiir::affine::AffineLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> indices(adaptor.getIndices());
    auto maybeExpandedMap = affine::expandAffineMap(rewriter, op.getLoc(),
                                                    op.getAffineMap(), indices);
    if (!maybeExpandedMap)
      return failure();

    auto coorOp = fir::CoordinateOp::create(
        rewriter, op.getLoc(),
        fir::ReferenceType::get(op.getResult().getType()), adaptor.getMemref(),
        *maybeExpandedMap);

    rewriter.replaceOpWithNewOp<fir::LoadOp>(op, coorOp.getResult());
    return success();
  }
};

class AffineStoreConversion
    : public OpConversionPattern<aiir::affine::AffineStoreOp> {
public:
  using OpConversionPattern<aiir::affine::AffineStoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(aiir::affine::AffineStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> indices(op.getIndices());
    auto maybeExpandedMap = affine::expandAffineMap(rewriter, op.getLoc(),
                                                    op.getAffineMap(), indices);
    if (!maybeExpandedMap)
      return failure();

    auto coorOp = fir::CoordinateOp::create(
        rewriter, op.getLoc(),
        fir::ReferenceType::get(op.getValueToStore().getType()),
        adaptor.getMemref(), *maybeExpandedMap);
    rewriter.replaceOpWithNewOp<fir::StoreOp>(op, adaptor.getValue(),
                                              coorOp.getResult());
    return success();
  }
};

class ConvertConversion : public aiir::OpRewritePattern<fir::ConvertOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  llvm::LogicalResult
  matchAndRewrite(fir::ConvertOp op,
                  aiir::PatternRewriter &rewriter) const override {
    if (aiir::isa<aiir::MemRefType>(op.getRes().getType())) {
      // due to index calculation moving to affine maps we still need to
      // add converts for sequence types this has a side effect of losing
      // some information about arrays with known dimensions by creating:
      // fir.convert %arg0 : (!fir.ref<!fir.array<5xi32>>) ->
      // !fir.ref<!fir.array<?xi32>>
      if (auto refTy =
              aiir::dyn_cast<fir::ReferenceType>(op.getValue().getType()))
        if (auto arrTy = aiir::dyn_cast<fir::SequenceType>(refTy.getEleTy())) {
          fir::SequenceType::Shape flatShape = {
              fir::SequenceType::getUnknownExtent()};
          auto flatArrTy = fir::SequenceType::get(flatShape, arrTy.getEleTy());
          auto flatTy = fir::ReferenceType::get(flatArrTy);
          rewriter.replaceOpWithNewOp<fir::ConvertOp>(op, flatTy,
                                                      op.getValue());
          return success();
        }
      rewriter.replaceOp(op, op.getValue());
    }
    return success();
  }
};

aiir::Type convertMemRef(aiir::MemRefType type) {
  return fir::SequenceType::get(SmallVector<int64_t>(type.getShape()),
                                type.getElementType());
}

class StdAllocConversion : public aiir::OpRewritePattern<memref::AllocOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  llvm::LogicalResult
  matchAndRewrite(memref::AllocOp op,
                  aiir::PatternRewriter &rewriter) const override {
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

    aiir::RewritePatternSet patterns(context);
    patterns.insert<ConvertConversion>(context);
    patterns.insert<AffineLoadConversion>(context);
    patterns.insert<AffineStoreConversion>(context);
    patterns.insert<StdAllocConversion>(context);
    aiir::ConversionTarget target(*context);
    target.addIllegalOp<memref::AllocOp>();
    target.addDynamicallyLegalOp<fir::ConvertOp>([](fir::ConvertOp op) {
      if (aiir::isa<aiir::MemRefType>(op.getRes().getType()))
        return false;
      return true;
    });
    target
        .addLegalDialect<FIROpsDialect, aiir::scf::SCFDialect,
                         aiir::arith::ArithDialect, aiir::func::FuncDialect>();

    if (aiir::failed(aiir::applyPartialConversion(function, target,
                                                  std::move(patterns)))) {
      aiir::emitError(aiir::UnknownLoc::get(context),
                      "error in converting affine dialect\n");
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<aiir::Pass> fir::createAffineDemotionPass() {
  return std::make_unique<AffineDialectDemotion>();
}
