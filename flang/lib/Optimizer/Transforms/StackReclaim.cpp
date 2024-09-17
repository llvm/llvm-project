//===- StackReclaim.cpp -- Insert stacksave/stackrestore in region --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Common/Fortran.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

namespace fir {
#define GEN_PASS_DEF_STACKRECLAIM
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace mlir;

namespace {

class StackReclaimPass : public fir::impl::StackReclaimBase<StackReclaimPass> {
public:
  using StackReclaimBase<StackReclaimPass>::StackReclaimBase;

  void runOnOperation() override;
};
} // namespace

void StackReclaimPass::runOnOperation() {
  auto *op = getOperation();
  fir::FirOpBuilder builder(op, fir::getKindMapping(op));

  op->walk([&](fir::DoLoopOp loopOp) {
    mlir::Location loc = loopOp.getLoc();

    if (!loopOp.getRegion().getOps<fir::AllocaOp>().empty()) {
      builder.setInsertionPointToStart(&loopOp.getRegion().front());
      mlir::Value sp = builder.genStackSave(loc);

      auto *terminator = loopOp.getRegion().back().getTerminator();
      builder.setInsertionPoint(terminator);
      builder.genStackRestore(loc, sp);
    }
  });
}
