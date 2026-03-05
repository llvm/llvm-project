//===-- CUFLaunchAttachAttr.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/CUF/CUFDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace fir {
#define GEN_PASS_DEF_CUFLAUNCHATTACHATTR
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace mlir;

namespace {

static constexpr llvm::StringRef cudaKernelInfix = "_cufk_";

class CUFGPUAttachAttrPattern
    : public OpRewritePattern<mlir::gpu::LaunchFuncOp> {
  using OpRewritePattern<mlir::gpu::LaunchFuncOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::gpu::LaunchFuncOp op,
                                PatternRewriter &rewriter) const override {
    op->setAttr(cuf::getProcAttrName(),
                cuf::ProcAttributeAttr::get(op.getContext(),
                                            cuf::ProcAttribute::Global));
    return mlir::success();
  }
};

struct CUFLaunchAttachAttr
    : public fir::impl::CUFLaunchAttachAttrBase<CUFLaunchAttachAttr> {

  void runOnOperation() override {
    auto *context = &this->getContext();

    mlir::RewritePatternSet patterns(context);
    patterns.add<CUFGPUAttachAttrPattern>(context);

    mlir::ConversionTarget target(*context);
    target.addIllegalOp<mlir::gpu::LaunchFuncOp>();
    target.addDynamicallyLegalOp<mlir::gpu::LaunchFuncOp>(
        [&](mlir::gpu::LaunchFuncOp op) -> bool {
          if (op.getKernelName().getValue().contains(cudaKernelInfix)) {
            if (op.getOperation()->getAttrOfType<cuf::ProcAttributeAttr>(
                    cuf::getProcAttrName()))
              return true;
            return false;
          }
          return true;
        });

    if (mlir::failed(mlir::applyPartialConversion(this->getOperation(), target,
                                                  std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "Pattern conversion failed\n");
      this->signalPassFailure();
    }
  }
};

} // end anonymous namespace
