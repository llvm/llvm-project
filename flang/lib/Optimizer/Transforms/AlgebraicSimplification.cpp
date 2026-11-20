//===- AlgebraicSimplification.cpp - Simplify algebraic expressions -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file defines a pass that applies algebraic simplifications
// to operations of Math/Complex/etc. dialects that are used by Flang.
// It is done as a Flang specific pass, because we may want to tune
// the parameters of the patterns for Fortran programs.
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/CUFCommon.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace fir {
#define GEN_PASS_DEF_ALGEBRAICSIMPLIFICATION
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace mlir;

namespace {
struct AlgebraicSimplification
    : public fir::impl::AlgebraicSimplificationBase<AlgebraicSimplification> {
  AlgebraicSimplification(const GreedyRewriteConfig &rewriteConfig,
                          bool keepDeviceComplexPowi = true) {
    config = rewriteConfig;
    this->keepDeviceComplexPowi = keepDeviceComplexPowi;
  }

  void runOnOperation() override;

  mlir::GreedyRewriteConfig config;
};
} // namespace

namespace {
/// The functions of a module whose whole body is OpenACC or CUDA Fortran
/// device code, collected on the first query.
class DeviceFuncs {
public:
  explicit DeviceFuncs(ModuleOp module) : module(module) {}

  bool contains(func::FuncOp func) {
    if (!funcs)
      funcs = cuf::collectFunctionsExecutingOnDevice(module);
    return funcs->contains(func);
  }

private:
  ModuleOp module;
  std::optional<llvm::SetVector<func::FuncOp>> funcs;
};
} // namespace

/// Return true if \p op is in OpenACC or CUDA Fortran device code.
static bool isInDeviceCode(Operation *op, DeviceFuncs &deviceFuncs) {
  if (fir::isInOffloadRegion(op))
    return true;
  auto func = op->getParentOfType<func::FuncOp>();
  return func && deviceFuncs.contains(func);
}

namespace {
/// Wrap a complex.powi pattern so that it does not apply to a power with a
/// negative constant exponent in device code. The strength reduction would
/// turn such a power into an inline complex division, whose intermediate
/// results can overflow although the power is representable. A power that is
/// kept can be lowered to a runtime call instead.
class DevicePowiGuard : public RewritePattern {
public:
  DevicePowiGuard(std::unique_ptr<RewritePattern> wrapped,
                  DeviceFuncs &deviceFuncs, MLIRContext *context)
      : RewritePattern(complex::PowiOp::getOperationName(),
                       wrapped->getBenefit(), context),
        wrapped(std::move(wrapped)), deviceFuncs(deviceFuncs) {
    setDebugName(this->wrapped->getDebugName());
    setHasBoundedRewriteRecursion(this->wrapped->hasBoundedRewriteRecursion());
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    APInt exponent;
    if (matchPattern(cast<complex::PowiOp>(op).getRhs(),
                     m_ConstantInt(&exponent)) &&
        exponent.isNegative() && isInDeviceCode(op, deviceFuncs))
      return rewriter.notifyMatchFailure(
          op, "negative constant exponent in device code");
    return wrapped->matchAndRewrite(op, rewriter);
  }

private:
  std::unique_ptr<RewritePattern> wrapped;
  DeviceFuncs &deviceFuncs;
};
} // namespace

void AlgebraicSimplification::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  populateMathAlgebraicSimplificationPatterns(patterns);

  DeviceFuncs deviceFuncs(module);
  if (keepDeviceComplexPowi) {
    OperationName powiName(complex::PowiOp::getOperationName(), context);
    for (std::unique_ptr<RewritePattern> &pattern :
         patterns.getNativePatterns())
      if (pattern->getRootKind() == powiName)
        pattern = std::make_unique<DevicePowiGuard>(std::move(pattern),
                                                    deviceFuncs, context);
  }
  (void)applyPatternsGreedily(module, std::move(patterns), config);
}

std::unique_ptr<mlir::Pass> fir::createAlgebraicSimplificationPass() {
  return std::make_unique<AlgebraicSimplification>(GreedyRewriteConfig());
}

std::unique_ptr<mlir::Pass>
fir::createAlgebraicSimplificationPass(const mlir::GreedyRewriteConfig &config,
                                       bool keepDeviceComplexPowi) {
  return std::make_unique<AlgebraicSimplification>(config,
                                                   keepDeviceComplexPowi);
}
