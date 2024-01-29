//===- FunctionAttr.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// \file
/// This is a generic pass for adding attributes to functions.
//===----------------------------------------------------------------------===//
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"

namespace fir {
#define GEN_PASS_DECL_FUNCTIONATTR
#define GEN_PASS_DEF_FUNCTIONATTR
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "func-attr"

namespace {

class FunctionAttrPass : public fir::impl::FunctionAttrBase<FunctionAttrPass> {
public:
  FunctionAttrPass(const fir::FunctionAttrOptions &options) {
    framePointerKind = options.framePointerKind;
    noInfsFPMath = options.noInfsFPMath;
    noNaNsFPMath = options.noNaNsFPMath;
    approxFuncFPMath = options.approxFuncFPMath;
    noSignedZerosFPMath = options.noSignedZerosFPMath;
    unsafeFPMath = options.unsafeFPMath;
  }
  FunctionAttrPass() {}
  void runOnOperation() override;
};

} // namespace

void FunctionAttrPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "=== Begin " DEBUG_TYPE " ===\n");
  mlir::func::FuncOp func = getOperation();

  LLVM_DEBUG(llvm::dbgs() << "Func-name:" << func.getSymName() << "\n");

  mlir::MLIRContext *context = &getContext();
  if (framePointerKind != mlir::LLVM::framePointerKind::FramePointerKind::None)
    func->setAttr("frame_pointer", mlir::LLVM::FramePointerKindAttr::get(
                                       context, framePointerKind));

  if (noInfsFPMath)
    func->setAttr("no_infs_fp_math", mlir::BoolAttr::get(context, true));
  if (noNaNsFPMath)
    func->setAttr("no_nans_fp_math", mlir::BoolAttr::get(context, true));
  if (approxFuncFPMath)
    func->setAttr("approx_func_fp_math", mlir::BoolAttr::get(context, true));
  if (noSignedZerosFPMath)
    func->setAttr("no_signed_zeros_fp_math",
                  mlir::BoolAttr::get(context, true));
  if (unsafeFPMath)
    func->setAttr("unsafe_fp_math", mlir::BoolAttr::get(context, true));

  LLVM_DEBUG(llvm::dbgs() << "=== End " DEBUG_TYPE " ===\n");
}

std::unique_ptr<mlir::Pass> fir::createFunctionAttrPass(
    fir::FunctionAttrTypes &functionAttr, bool noInfsFPMath, bool noNaNsFPMath,
    bool approxFuncFPMath, bool noSignedZerosFPMath, bool unsafeFPMath) {
  FunctionAttrOptions opts;
  // Frame pointer
  opts.framePointerKind = functionAttr.framePointerKind;
  opts.noInfsFPMath = noInfsFPMath;
  opts.noNaNsFPMath = noNaNsFPMath;
  opts.approxFuncFPMath = approxFuncFPMath;
  opts.noSignedZerosFPMath = noSignedZerosFPMath;
  opts.unsafeFPMath = unsafeFPMath;

  return std::make_unique<FunctionAttrPass>(opts);
}

std::unique_ptr<mlir::Pass> fir::createFunctionAttrPass() {
  return std::make_unique<FunctionAttrPass>();
}
