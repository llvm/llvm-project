//===-- CUFFUnctionRewrite.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/CodeGen/TypeConverter.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/DataLayout.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include <string_view>

#define DEBUG_TYPE "flang-cuf-function-rewrite"

namespace fir {
#define GEN_PASS_DEF_CUFFUNCTIONREWRITE
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace mlir;

namespace {

using genFunctionType =
    std::function<mlir::Value(mlir::PatternRewriter &, fir::CallOp op)>;

class CallConversion : public OpRewritePattern<fir::CallOp> {
public:
  CallConversion(MLIRContext *context)
      : OpRewritePattern<fir::CallOp>(context) {}

  LogicalResult
  matchAndRewrite(fir::CallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto callee = op.getCallee();
    if (!callee)
      return failure();
    auto name = callee->getRootReference().getValue();

    if (genMappings_.contains(name)) {
      auto fct = genMappings_.find(name);
      mlir::Value result = fct->second(rewriter, op);
      if (result)
        rewriter.replaceOp(op, result);
      else
        rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }

private:
  static mlir::Value genOnDevice(mlir::PatternRewriter &rewriter,
                                 fir::CallOp op) {
    assert(op.getArgs().size() == 0 && "expect 0 arguments");
    mlir::Location loc = op.getLoc();
    unsigned inGPUMod = op->getParentOfType<gpu::GPUModuleOp>() ? 1 : 0;
    mlir::Type i1Ty = rewriter.getIntegerType(1);
    mlir::Value t = mlir::arith::ConstantOp::create(
        rewriter, loc, i1Ty, rewriter.getIntegerAttr(i1Ty, inGPUMod));
    return fir::ConvertOp::create(rewriter, loc, op.getResult(0).getType(), t);
  }

  const llvm::StringMap<genFunctionType> genMappings_ = {
      {"on_device", &genOnDevice}};
};

class CUFFunctionRewrite
    : public fir::impl::CUFFunctionRewriteBase<CUFFunctionRewrite> {
public:
  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.insert<CallConversion>(patterns.getContext());

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(ctx),
                      "error in CUFFunctionRewrite op conversion\n");
      signalPassFailure();
    }
  }
};

} // namespace
