//===-- RaiseToAffine.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/CommandLine.h"

/// disable FIR to affine dialect conversion
static llvm::cl::opt<bool>
    disableAffinePromo("disable-affine-promotion",
                       llvm::cl::desc("disable FIR to Affine pass"),
                       llvm::cl::init(false));

using namespace fir;

namespace {

template <typename FROM>
class OpRewrite : public mlir::RewritePattern {
public:
  explicit OpRewrite(mlir::MLIRContext *ctx)
      : RewritePattern(FROM::getOperationName(), 1, ctx) {}
};

/// Convert `fir.loop` to `affine.for`
class AffineLoopConv : public OpRewrite<LoopOp> {
public:
  using OpRewrite::OpRewrite;
};

/// Convert `fir.where` to `affine.if`
class AffineWhereConv : public OpRewrite<WhereOp> {
public:
  using OpRewrite::OpRewrite;
};

/// Promote fir.loop and fir.where to affine.for and affine.if, in the cases
/// where such a promotion is possible.
class AffineDialectPromotion
  : public AffineDialectPromotionBase<AffineDialectPromotion> {
public:
  void runOnFunction() override {
    if (disableAffinePromo)
      return;

    auto *context = &getContext();
    mlir::OwningRewritePatternList patterns;
    patterns.insert<AffineLoopConv, AffineWhereConv>(context);
    mlir::ConversionTarget target = *context;
    target.addLegalDialect<mlir::AffineDialect, FIROpsDialect,
                           mlir::scf::SCFDialect,
                           mlir::StandardOpsDialect>();
    // target.addDynamicallyLegalOp<LoopOp, WhereOp>();

    // apply the patterns
    if (mlir::failed(mlir::applyPartialConversion(getFunction(), target,
                                                  std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "error in converting to affine dialect\n");
      signalPassFailure();
    }
  }
};

} // namespace

/// Convert FIR loop constructs to the Affine dialect
std::unique_ptr<mlir::Pass> fir::createPromoteToAffinePass() {
  return std::make_unique<AffineDialectPromotion>();
}
