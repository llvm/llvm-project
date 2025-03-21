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
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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
  // TODO: support preferInlineImplementation.
  bool doInline = options.preferInlineImplementation && false;
  if (!doInline) {
    // Generate Fortran runtime call.
    mlir::Value result;
    if (op.getInnermost()) {
      mlir::Value one =
          builder.createIntegerConstant(loc, builder.getI32Type(), 1);
      result =
          fir::runtime::genIsContiguousUpTo(builder, loc, op.getBox(), one);
    } else {
      result = fir::runtime::genIsContiguous(builder, loc, op.getBox());
    }
    result = builder.createConvert(loc, op.getType(), result);
    rewriter.replaceOp(op, result);
    return mlir::success();
  }

  // Generate inline implementation.
  TODO(loc, "inline IsContiguousBoxOp");
  return mlir::failure();
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

void SimplifyFIROperationsPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  mlir::MLIRContext &context = getContext();
  mlir::RewritePatternSet patterns(&context);
  fir::populateSimplifyFIROperationsPatterns(patterns,
                                             preferInlineImplementation);
  mlir::GreedyRewriteConfig config;
  config.enableRegionSimplification = mlir::GreedySimplifyRegionLevel::Disabled;

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
}
