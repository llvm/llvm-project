//===-- CUFFunctionRewrite.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Support/DataLayout.h"
#include "flang/Optimizer/Support/Utils.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"

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
  CallConversion(MLIRContext *context, bool deferAccRoutines)
      : OpRewritePattern<fir::CallOp>(context),
        deferAccRoutines_(deferAccRoutines) {}

  LogicalResult
  matchAndRewrite(fir::CallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto callee = op.getCallee();
    if (!callee)
      return failure();

    // Match on the callee's Fortran leaf name rather than on its symbol name so
    // this does not depend on the target's name-mangling convention or on where
    // in the pipeline the pass runs. getPresentableFunctionName restores the
    // original name saved by external-name conversion and returns the
    // deconstructed leaf name.
    auto func = mlir::dyn_cast_or_null<mlir::FunctionOpInterface>(
        mlir::SymbolTable::lookupNearestSymbolFrom(op, *callee));
    if (!func)
      return failure();

    auto fct = genMappings_.find(fir::getPresentableFunctionName(func));
    if (fct == genMappings_.end())
      return failure();

    // Only rewrite a compiler-provided declaration, never a user-defined
    // procedure that happens to share the name.
    if (!func.isExternal())
      return failure();

    // Defer folding in the host copy of an OpenACC routine. Device
    // specialization later clones the host body to build the device routine, so
    // folding it to the host value now would bake that value into the device
    // clone. A later run (after specialization) folds each copy in its own
    // host/device context. Calls already inside a gpu.module are device copies
    // and are always safe to fold.
    if (deferAccRoutines_ && !op->getParentOfType<gpu::GPUModuleOp>()) {
      if (auto enclosing = op->getParentOfType<mlir::FunctionOpInterface>())
        if (mlir::acc::isAccRoutine(enclosing))
          return failure();
    }

    mlir::Value result = fct->second(rewriter, op);
    if (!result)
      return failure();
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  static mlir::Value genOnDevice(mlir::PatternRewriter &rewriter,
                                 fir::CallOp op) {
    // Only fold calls that match the intrinsic's shape: no arguments and a
    // single logical result.
    if (!op.getArgs().empty() || op.getNumResults() != 1)
      return {};
    mlir::Type resTy = op.getResult(0).getType();
    if (!mlir::isa<fir::LogicalType>(resTy))
      return {};
    mlir::Location loc = op.getLoc();
    unsigned inGPUMod = op->getParentOfType<gpu::GPUModuleOp>() ? 1 : 0;
    mlir::Type i1Ty = rewriter.getIntegerType(1);
    mlir::Value t = mlir::arith::ConstantOp::create(
        rewriter, loc, i1Ty, rewriter.getIntegerAttr(i1Ty, inGPUMod));
    return fir::ConvertOp::create(rewriter, loc, resTy, t);
  }

  // Recognized by Fortran leaf name; see matchAndRewrite for how the leaf name
  // is recovered independently of external name mangling.
  const llvm::StringMap<genFunctionType> genMappings_ = {
      {"on_device", &genOnDevice}};

  bool deferAccRoutines_ = false;
};

class CUFFunctionRewrite
    : public fir::impl::CUFFunctionRewriteBase<CUFFunctionRewrite> {
public:
  using CUFFunctionRewriteBase::CUFFunctionRewriteBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.insert<CallConversion>(patterns.getContext(), deferAccRoutines);

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(ctx),
                      "error in CUFFunctionRewrite op conversion\n");
      signalPassFailure();
    }
  }
};

} // namespace
