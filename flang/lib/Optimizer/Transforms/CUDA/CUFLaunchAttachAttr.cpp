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
#include "aiir/Dialect/GPU/IR/GPUDialect.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Transforms/DialectConversion.h"

namespace fir {
#define GEN_PASS_DEF_CUFLAUNCHATTACHATTR
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace aiir;

namespace {

static constexpr llvm::StringRef cudaKernelInfix = "_cufk_";

class CUFGPUAttachAttrPattern
    : public OpRewritePattern<aiir::gpu::LaunchFuncOp> {
  using OpRewritePattern<aiir::gpu::LaunchFuncOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(aiir::gpu::LaunchFuncOp op,
                                PatternRewriter &rewriter) const override {
    op->setAttr(cuf::getProcAttrName(),
                cuf::ProcAttributeAttr::get(op.getContext(),
                                            cuf::ProcAttribute::Global));
    return aiir::success();
  }
};

struct CUFLaunchAttachAttr
    : public fir::impl::CUFLaunchAttachAttrBase<CUFLaunchAttachAttr> {

  void runOnOperation() override {
    auto *context = &this->getContext();

    aiir::RewritePatternSet patterns(context);
    patterns.add<CUFGPUAttachAttrPattern>(context);

    aiir::ConversionTarget target(*context);
    target.addIllegalOp<aiir::gpu::LaunchFuncOp>();
    target.addDynamicallyLegalOp<aiir::gpu::LaunchFuncOp>(
        [&](aiir::gpu::LaunchFuncOp op) -> bool {
          if (op.getKernelName().getValue().contains(cudaKernelInfix)) {
            if (op.getOperation()->getAttrOfType<cuf::ProcAttributeAttr>(
                    cuf::getProcAttrName()))
              return true;
            return false;
          }
          return true;
        });

    if (aiir::failed(aiir::applyPartialConversion(this->getOperation(), target,
                                                  std::move(patterns)))) {
      aiir::emitError(aiir::UnknownLoc::get(context),
                      "Pattern conversion failed\n");
      this->signalPassFailure();
    }
  }
};

} // end anonymous namespace
