//===-- CUFLaunchAttachAttr.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/CUF/Attributes/CUFAttr.h"
#include "flang/Optimizer/Dialect/CUF/CUFDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Pass/Pass.h"

namespace fir {
#define GEN_PASS_DEF_CUFLAUNCHATTACHATTR
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace mlir;

namespace {

static constexpr llvm::StringRef cudaKernelInfix = "_cufk_";

struct CUFLaunchAttachAttr
    : public fir::impl::CUFLaunchAttachAttrBase<CUFLaunchAttachAttr> {

  void runOnOperation() override {
    getOperation()->walk([](gpu::LaunchFuncOp op) {
      if (!op.getKernelName().getValue().contains(cudaKernelInfix))
        return;
      if (op->getAttrOfType<cuf::ProcAttributeAttr>(cuf::getProcAttrName()))
        return;
      op->setAttr(cuf::getProcAttrName(),
                  cuf::ProcAttributeAttr::get(op.getContext(),
                                              cuf::ProcAttribute::Global));
    });
  }
};

} // end anonymous namespace
