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

#include "flang/ISO_Fortran_binding_wrapper.h"
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
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

namespace fir {
#define GEN_PASS_DECL_VSCALEATTR
#define GEN_PASS_DEF_VSCALEATTR
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "vscale-attr"

namespace {

class VScaleAttrPass : public fir::impl::VScaleAttrBase<VScaleAttrPass> {
public:
  VScaleAttrPass(const fir::VScaleAttrOptions &options) {
    vscaleRange = options.vscaleRange;
  }
  VScaleAttrPass() {}
  void runOnOperation() override;
};

} // namespace

void VScaleAttrPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "=== Begin " DEBUG_TYPE " ===\n");
  mlir::func::FuncOp func = getOperation();

  LLVM_DEBUG(llvm::dbgs() << "Func-name:" << func.getSymName() << "\n");

  auto context = &getContext();

  auto intTy = mlir::IntegerType::get(context, 32);

  assert(vscaleRange.first && "VScaleRange minimum should be non-zero");

  func->setAttr("vscale_range",
                mlir::LLVM::VScaleRangeAttr::get(
                    context, mlir::IntegerAttr::get(intTy, vscaleRange.first),
                    mlir::IntegerAttr::get(intTy, vscaleRange.second)));

  LLVM_DEBUG(llvm::dbgs() << "=== End " DEBUG_TYPE " ===\n");
}

std::unique_ptr<mlir::Pass>
fir::createVScaleAttrPass(std::pair<unsigned, unsigned> vscaleAttr) {
  VScaleAttrOptions opts;
  opts.vscaleRange = vscaleAttr;
  return std::make_unique<VScaleAttrPass>(opts);
}

std::unique_ptr<mlir::Pass> fir::createVScaleAttrPass() {
  return std::make_unique<VScaleAttrPass>();
}
