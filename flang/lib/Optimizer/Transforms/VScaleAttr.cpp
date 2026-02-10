//===- VScaleAttr.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// \file
/// This pass adds a `vscale_range` attribute to function definitions.
/// The attribute is used for scalable vector operations on Arm processors
/// and should only be run on processors that support this feature. [It is
/// likely harmless to run it on something else, but it is also not valuable].
//===----------------------------------------------------------------------===//

#include "flang/Common/ISO_Fortran_binding_wrapper.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/Inquiry.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <mlir/IR/Diagnostics.h>

namespace fir {
#define GEN_PASS_DEF_VSCALEATTR
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "vscale-attr"

namespace {

class VScaleAttrPass : public fir::impl::VScaleAttrBase<VScaleAttrPass> {
public:
  VScaleAttrPass(const fir::VScaleAttrOptions &options) {
    vscaleMin = options.vscaleMin;
    vscaleMax = options.vscaleMax;
  }
  VScaleAttrPass() {}
  void runOnOperation() override;
};

} // namespace

void VScaleAttrPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "=== Begin " DEBUG_TYPE " ===\n");
  mlir::func::FuncOp func = getOperation();

  LLVM_DEBUG(llvm::dbgs() << "Func-name:" << func.getSymName() << "\n");

  if (!llvm::isPowerOf2_32(vscaleMin)) {
    func->emitError(
        "VScaleAttr: vscaleMin has to be a power-of-two greater than 0\n");
    return signalPassFailure();
  }

  if (vscaleMax != 0 &&
      (!llvm::isPowerOf2_32(vscaleMax) || (vscaleMin > vscaleMax))) {
    func->emitError("VScaleAttr: vscaleMax has to be a power-of-two "
                    "greater-than-or-equal to vscaleMin or 0 to signify "
                    "an unbounded maximum\n");
    return signalPassFailure();
  }

  auto context = &getContext();

  auto intTy = mlir::IntegerType::get(context, 32);

  func->setAttr("vscale_range",
                mlir::LLVM::VScaleRangeAttr::get(
                    context, mlir::IntegerAttr::get(intTy, vscaleMin),
                    mlir::IntegerAttr::get(intTy, vscaleMax)));

  LLVM_DEBUG(llvm::dbgs() << "=== End " DEBUG_TYPE " ===\n");
}
