//===- PropagateFortranVariableAttributes.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file defines a pass that propagates FortranVariableFlagsAttr
/// attributes through HLFIR. For example, it can set contiguous attribute
/// on hlfir.designate that produces a contiguous slice of a contiguous
/// Fortran array. This pass can be applied multiple times to expose
/// more Fortran attributes, e.g. after inlining and constant propagation.
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "llvm/ADT/TypeSwitch.h"

namespace hlfir {
#define GEN_PASS_DEF_PROPAGATEFORTRANVARIABLEATTRIBUTES
#include "flang/Optimizer/HLFIR/Passes.h.inc"
} // namespace hlfir

#define DEBUG_TYPE "propagate-fortran-attrs"

namespace {
class PropagateFortranVariableAttributes
    : public hlfir::impl::PropagateFortranVariableAttributesBase<
          PropagateFortranVariableAttributes> {
public:
  using PropagateFortranVariableAttributesBase<
      PropagateFortranVariableAttributes>::
      PropagateFortranVariableAttributesBase;
  void runOnOperation() override;
};

class Propagator {
public:
  void process(mlir::Operation *op);

private:
  static bool isContiguous(mlir::Operation *op) {
    // Treat data allocations as contiguous, so that we can propagate
    // the continuity from them. Allocations of fir.box must not be treated
    // as contiguous.
    if (mlir::isa<fir::AllocaOp, fir::AllocMemOp>(op) &&
        !mlir::isa<fir::BaseBoxType>(
            fir::unwrapRefType(op->getResult(0).getType())))
      return true;
    auto varOp = mlir::dyn_cast<fir::FortranVariableOpInterface>(op);
    if (!varOp)
      return false;
    return hlfir::Entity{varOp}.isSimplyContiguous();
  }

  static void setContiguousAttr(fir::FortranVariableOpInterface op);
};
} // namespace

void Propagator::setContiguousAttr(fir::FortranVariableOpInterface op) {
  LLVM_DEBUG(llvm::dbgs() << "Setting continuity for:\n" << op << "\n");
  fir::FortranVariableFlagsEnum attrs =
      op.getFortranAttrs().value_or(fir::FortranVariableFlagsEnum::None);
  attrs = attrs | fir::FortranVariableFlagsEnum::contiguous;
  op.setFortranAttrs(attrs);
}

void Propagator::process(mlir::Operation *op) {
  if (!isContiguous(op))
    return;
  llvm::SmallVector<mlir::Operation *> workList{op};
  while (!workList.empty()) {
    mlir::Operation *current = workList.pop_back_val();
    LLVM_DEBUG(llvm::dbgs() << "Propagating continuity from operation:\n"
                            << *current << "\n");

    for (mlir::OpOperand &use : current->getUses()) {
      mlir::Operation *useOp = use.getOwner();
      if (auto varOp = mlir::dyn_cast<fir::FortranVariableOpInterface>(useOp)) {
        // If the user is not currently contiguous, set the contiguous
        // attribute and skip it. The propagation will pick it up later.
        mlir::Value memref;
        mlir::TypeSwitch<mlir::Operation *, void>(useOp)
            .Case<hlfir::DeclareOp, hlfir::DesignateOp>(
                [&](auto op) { memref = op.getMemref(); })
            .Default([&](auto op) {});

        if (memref == use.get() && !isContiguous(varOp)) {
          // Make additional checks for hlfir.designate.
          if (auto designateOp = mlir::dyn_cast<hlfir::DesignateOp>(useOp))
            if (!hlfir::designatePreservesContinuity(designateOp))
              continue;

          setContiguousAttr(varOp);
        }
        continue;
      }
      mlir::TypeSwitch<mlir::Operation *, void>(useOp)
          .Case(
              [&](fir::ConvertOp op) { workList.push_back(op.getOperation()); })
          .Case([&](fir::EmboxOp op) {
            if (op.getMemref() == use.get())
              workList.push_back(op.getOperation());
          })
          .Case([&](fir::ReboxOp op) {
            if (op.getBox() == use.get() && fir::reboxPreservesContinuity(op))
              workList.push_back(op.getOperation());
          });
    }
  }
}

void PropagateFortranVariableAttributes::runOnOperation() {
  mlir::Operation *rootOp = getOperation();
  mlir::MLIRContext *context = &getContext();
  mlir::RewritePatternSet patterns(context);
  Propagator propagator;
  rootOp->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    propagator.process(op);
    return mlir::WalkResult::advance();
  });
}
