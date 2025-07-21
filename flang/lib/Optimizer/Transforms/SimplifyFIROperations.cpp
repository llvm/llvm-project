//===- SimplifyFIROperations.cpp -- simplify complex FIR operations  ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// \file
/// This pass transforms some FIR operations into their equivalent
/// implementations using other FIR operations. The transformation
/// can legally use SCF dialect and generate Fortran runtime calls.
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/Inquiry.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <optional>

namespace fir {
#define GEN_PASS_DEF_SIMPLIFYFIROPERATIONS
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "flang-simplify-fir-operations"

namespace {
/// Pass runner.
class SimplifyFIROperationsPass
    : public fir::impl::SimplifyFIROperationsBase<SimplifyFIROperationsPass> {
public:
  using fir::impl::SimplifyFIROperationsBase<
      SimplifyFIROperationsPass>::SimplifyFIROperationsBase;

  void runOnOperation() override final;
};

/// Base class for all conversions holding the pass options.
template <typename Op>
class ConversionBase : public mlir::OpRewritePattern<Op> {
public:
  using mlir::OpRewritePattern<Op>::OpRewritePattern;

  template <typename... Args>
  ConversionBase(mlir::MLIRContext *context, Args &&...args)
      : mlir::OpRewritePattern<Op>(context),
        options{std::forward<Args>(args)...} {}

  mlir::LogicalResult matchAndRewrite(Op,
                                      mlir::PatternRewriter &) const override;

protected:
  fir::SimplifyFIROperationsOptions options;
};

/// fir::IsContiguousBoxOp converter.
using IsContiguousBoxCoversion = ConversionBase<fir::IsContiguousBoxOp>;

/// fir::BoxTotalElementsOp converter.
using BoxTotalElementsConversion = ConversionBase<fir::BoxTotalElementsOp>;
} // namespace

/// Generate a call to IsContiguous/IsContiguousUpTo function or an inline
/// sequence reading extents/strides from the box and checking them.
/// This conversion may produce fir.box_elesize and a loop (for assumed
/// rank).
template <>
mlir::LogicalResult IsContiguousBoxCoversion::matchAndRewrite(
    fir::IsContiguousBoxOp op, mlir::PatternRewriter &rewriter) const {
  mlir::Location loc = op.getLoc();
  fir::FirOpBuilder builder(rewriter, op.getOperation());
  mlir::Value box = op.getBox();

  if (options.preferInlineImplementation) {
    auto boxType = mlir::cast<fir::BaseBoxType>(box.getType());
    unsigned rank = fir::getBoxRank(boxType);

    // If rank is one, or 'innermost' attribute is set and
    // it is not a scalar, then generate a simple comparison
    // for the leading dimension: (stride == elem_size || extent == 0).
    //
    // The scalar cases are supposed to be optimized by the canonicalization.
    if (rank == 1 || (op.getInnermost() && rank > 0)) {
      mlir::Type idxTy = builder.getIndexType();
      auto eleSize = fir::BoxEleSizeOp::create(builder, loc, idxTy, box);
      mlir::Value zero = fir::factory::createZeroValue(builder, loc, idxTy);
      auto dimInfo =
          fir::BoxDimsOp::create(builder, loc, idxTy, idxTy, idxTy, box, zero);
      mlir::Value stride = dimInfo.getByteStride();
      mlir::Value pred1 = mlir::arith::CmpIOp::create(
          builder, loc, mlir::arith::CmpIPredicate::eq, eleSize, stride);
      mlir::Value extent = dimInfo.getExtent();
      mlir::Value pred2 = mlir::arith::CmpIOp::create(
          builder, loc, mlir::arith::CmpIPredicate::eq, extent, zero);
      mlir::Value result =
          mlir::arith::OrIOp::create(builder, loc, pred1, pred2);
      result = builder.createConvert(loc, op.getType(), result);
      rewriter.replaceOp(op, result);
      return mlir::success();
    }
    // TODO: support arrays with multiple dimensions.
  }

  // Generate Fortran runtime call.
  mlir::Value result;
  if (op.getInnermost()) {
    mlir::Value one =
        builder.createIntegerConstant(loc, builder.getI32Type(), 1);
    result = fir::runtime::genIsContiguousUpTo(builder, loc, box, one);
  } else {
    result = fir::runtime::genIsContiguous(builder, loc, box);
  }
  result = builder.createConvert(loc, op.getType(), result);
  rewriter.replaceOp(op, result);
  return mlir::success();
}

/// Generate a call to Size runtime function or an inline
/// sequence reading extents from the box an multiplying them.
/// This conversion may produce a loop (for assumed rank).
template <>
mlir::LogicalResult BoxTotalElementsConversion::matchAndRewrite(
    fir::BoxTotalElementsOp op, mlir::PatternRewriter &rewriter) const {
  mlir::Location loc = op.getLoc();
  fir::FirOpBuilder builder(rewriter, op.getOperation());
  // TODO: support preferInlineImplementation.
  // Reading the extent from the box for 1D arrays probably
  // results in less code than the call, so we can always
  // inline it.
  bool doInline = options.preferInlineImplementation && false;
  if (!doInline) {
    // Generate Fortran runtime call.
    mlir::Value result = fir::runtime::genSize(builder, loc, op.getBox());
    result = builder.createConvert(loc, op.getType(), result);
    rewriter.replaceOp(op, result);
    return mlir::success();
  }

  // Generate inline implementation.
  TODO(loc, "inline BoxTotalElementsOp");
  return mlir::failure();
}

class DoConcurrentConversion
    : public mlir::OpRewritePattern<fir::DoConcurrentOp> {
  /// Looks up from the operation from and returns the LocalitySpecifierOp with
  /// name symbolName
  static fir::LocalitySpecifierOp
  findLocalizer(mlir::Operation *from, mlir::SymbolRefAttr symbolName) {
    fir::LocalitySpecifierOp localizer =
        mlir::SymbolTable::lookupNearestSymbolFrom<fir::LocalitySpecifierOp>(
            from, symbolName);
    assert(localizer && "localizer not found in the symbol table");
    return localizer;
  }

public:
  using mlir::OpRewritePattern<fir::DoConcurrentOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(fir::DoConcurrentOp doConcurentOp,
                  mlir::PatternRewriter &rewriter) const override {
    assert(doConcurentOp.getRegion().hasOneBlock());
    mlir::Block &wrapperBlock = doConcurentOp.getRegion().getBlocks().front();
    auto loop =
        mlir::cast<fir::DoConcurrentLoopOp>(wrapperBlock.getTerminator());
    assert(loop.getRegion().hasOneBlock());
    mlir::Block &loopBlock = loop.getRegion().getBlocks().front();

    // Handle localization
    if (!loop.getLocalVars().empty()) {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&loop.getRegion().front());

      std::optional<mlir::ArrayAttr> localSyms = loop.getLocalSyms();

      for (auto localInfo : llvm::zip_equal(
               loop.getLocalVars(), loop.getRegionLocalArgs(), *localSyms)) {
        mlir::Value localVar = std::get<0>(localInfo);
        mlir::BlockArgument localArg = std::get<1>(localInfo);
        mlir::Attribute localizerSym = std::get<2>(localInfo);
        mlir::SymbolRefAttr localizerName =
            llvm::cast<mlir::SymbolRefAttr>(localizerSym);
        fir::LocalitySpecifierOp localizer = findLocalizer(loop, localizerName);

        // TODO Should this be a heap allocation instead? For now, we allocate
        // on the stack for each loop iteration.
        mlir::Value localAlloc =
            fir::AllocaOp::create(rewriter, loop.getLoc(), localizer.getType());

        auto cloneLocalizerRegion = [&](mlir::Region &region,
                                        mlir::ValueRange regionArgs,
                                        mlir::Block::iterator insertionPoint) {
          // It is reasonable to make this assumption since, at this stage,
          // control-flow ops are not converted yet. Therefore, things like `if`
          // conditions will still be represented by their encapsulating `fir`
          // dialect ops.
          assert(region.hasOneBlock() &&
                 "Expected localizer region to have a single block.");
          mlir::OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(rewriter.getInsertionBlock(),
                                     insertionPoint);
          mlir::IRMapping mapper;
          mapper.map(region.getArguments(), regionArgs);
          for (mlir::Operation &op : region.front().without_terminator())
            (void)rewriter.clone(op, mapper);
        };

        if (!localizer.getInitRegion().empty())
          cloneLocalizerRegion(localizer.getInitRegion(), {localVar, localArg},
                               rewriter.getInsertionPoint());

        if (localizer.getLocalitySpecifierType() ==
            fir::LocalitySpecifierType::LocalInit)
          cloneLocalizerRegion(localizer.getCopyRegion(), {localVar, localArg},
                               rewriter.getInsertionPoint());

        if (!localizer.getDeallocRegion().empty())
          cloneLocalizerRegion(localizer.getDeallocRegion(), {localArg},
                               rewriter.getInsertionBlock()->end());

        rewriter.replaceAllUsesWith(localArg, localAlloc);
      }

      loop.getRegion().front().eraseArguments(loop.getNumInductionVars(),
                                              loop.getNumLocalOperands());
      loop.getLocalVarsMutable().clear();
      loop.setLocalSymsAttr(nullptr);
    }

    for (auto [reduceVar, reduceArg] :
         llvm::zip_equal(loop.getReduceVars(), loop.getRegionReduceArgs()))
      rewriter.replaceAllUsesWith(reduceArg, reduceVar);

    // Collect iteration variable(s) allocations so that we can move them
    // outside the `fir.do_concurrent` wrapper.
    llvm::SmallVector<mlir::Operation *> opsToMove;
    for (mlir::Operation &op : llvm::drop_end(wrapperBlock))
      opsToMove.push_back(&op);

    fir::FirOpBuilder firBuilder(
        rewriter, doConcurentOp->getParentOfType<mlir::ModuleOp>());
    auto *allocIt = firBuilder.getAllocaBlock();

    for (mlir::Operation *op : llvm::reverse(opsToMove))
      rewriter.moveOpBefore(op, allocIt, allocIt->begin());

    rewriter.setInsertionPointAfter(doConcurentOp);
    fir::DoLoopOp innermostUnorderdLoop;
    mlir::SmallVector<mlir::Value> ivArgs;

    for (auto [lb, ub, st, iv] :
         llvm::zip_equal(loop.getLowerBound(), loop.getUpperBound(),
                         loop.getStep(), *loop.getLoopInductionVars())) {
      innermostUnorderdLoop = fir::DoLoopOp::create(
          rewriter, doConcurentOp.getLoc(), lb, ub, st,
          /*unordred=*/true, /*finalCountValue=*/false,
          /*iterArgs=*/mlir::ValueRange{}, loop.getReduceVars(),
          loop.getReduceAttrsAttr());
      ivArgs.push_back(innermostUnorderdLoop.getInductionVar());
      rewriter.setInsertionPointToStart(innermostUnorderdLoop.getBody());
    }

    loop.getRegion().front().eraseArguments(loop.getNumInductionVars() +
                                                loop.getNumLocalOperands(),
                                            loop.getNumReduceOperands());

    rewriter.inlineBlockBefore(
        &loopBlock, innermostUnorderdLoop.getBody()->getTerminator(), ivArgs);
    rewriter.eraseOp(doConcurentOp);
    return mlir::success();
  }
};

void SimplifyFIROperationsPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  mlir::MLIRContext &context = getContext();
  mlir::RewritePatternSet patterns(&context);
  fir::populateSimplifyFIROperationsPatterns(patterns,
                                             preferInlineImplementation);
  mlir::GreedyRewriteConfig config;
  config.setRegionSimplificationLevel(
      mlir::GreedySimplifyRegionLevel::Disabled);

  if (mlir::failed(
          mlir::applyPatternsGreedily(module, std::move(patterns), config))) {
    mlir::emitError(module.getLoc(), DEBUG_TYPE " pass failed");
    signalPassFailure();
  }
}

void fir::populateSimplifyFIROperationsPatterns(
    mlir::RewritePatternSet &patterns, bool preferInlineImplementation) {
  patterns.insert<IsContiguousBoxCoversion, BoxTotalElementsConversion>(
      patterns.getContext(), preferInlineImplementation);
  patterns.insert<DoConcurrentConversion>(patterns.getContext());
}
