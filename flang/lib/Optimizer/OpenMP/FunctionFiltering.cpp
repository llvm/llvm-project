//===- FunctionFiltering.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements transforms to filter out functions intended for the host
// when compiling for the device and vice versa.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/OpenMP/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

namespace flangomp {
#define GEN_PASS_DEF_FUNCTIONFILTERINGPASS
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

using namespace mlir;

/// This function triggers TODO errors and halts compilation if it detects
/// patterns representing unimplemented features.
///
/// It exclusively checks situations that cannot be detected after all of the
/// MLIR pipeline has ran (i.e. at the MLIR to LLVM IR translation stage, where
/// the preferred location for these types of checks is), and it only checks for
/// features that have not been implemented for target offload, but are
/// supported on host execution.
void checkDeviceImplementationStatus(
    omp::OffloadModuleInterface offloadModule) {
  if (!offloadModule.getIsGPU())
    return;

  offloadModule->walk<WalkOrder::PreOrder>([&](omp::DeclareReductionOp redOp) {
    if (redOp.symbolKnownUseEmpty(offloadModule))
      return WalkResult::advance();

    if (!redOp.getByrefElementType())
      return WalkResult::advance();

    auto seqTy =
        mlir::dyn_cast<fir::SequenceType>(*redOp.getByrefElementType());

    bool isByRefReductionSupported =
        !seqTy || !fir::sequenceWithNonConstantShape(seqTy);

    if (!isByRefReductionSupported) {
      TODO(redOp.getLoc(),
           "Reduction of dynamically-shaped arrays are not supported yet "
           "on the GPU.");
    }

    return WalkResult::advance();
  });
}

namespace {
class FunctionFilteringPass
    : public flangomp::impl::FunctionFilteringPassBase<FunctionFilteringPass> {
public:
  FunctionFilteringPass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    OpBuilder opBuilder(context);
    auto op = dyn_cast<omp::OffloadModuleInterface>(getOperation());
    if (!op || !op.getIsTargetDevice())
      return;

    op->walk<WalkOrder::PreOrder>([&](func::FuncOp funcOp) {
      // Do not filter functions with target regions inside, because they have
      // to be available for both host and device so that regular and reverse
      // offloading can be supported.
      bool hasTargetRegion =
          funcOp
              ->walk<WalkOrder::PreOrder>(
                  [&](omp::TargetOp) { return WalkResult::interrupt(); })
              .wasInterrupted();

      omp::DeclareTargetDeviceType declareType =
          omp::DeclareTargetDeviceType::host;
      auto declareTargetOp =
          dyn_cast<omp::DeclareTargetInterface>(funcOp.getOperation());
      if (declareTargetOp && declareTargetOp.isDeclareTarget())
        declareType = declareTargetOp.getDeclareTargetDeviceType();

      // Filtering a function here means deleting it if it doesn't contain a
      // target region. Else we explicitly set the omp.declare_target
      // attribute. The second stage of function filtering at the MLIR to LLVM
      // IR translation level will remove functions that contain the target
      // region from the generated llvm IR.
      if (declareType == omp::DeclareTargetDeviceType::host) {
        SymbolTable::UseRange funcUses = *funcOp.getSymbolUses(op);
        for (SymbolTable::SymbolUse use : funcUses) {
          Operation *callOp = use.getUser();
          if (auto internalFunc = mlir::dyn_cast<func::FuncOp>(callOp)) {
            // Do not delete internal procedures holding the symbol of their
            // Fortran host procedure as attribute.
            internalFunc->removeAttr(fir::getHostSymbolAttrName());
            // Set public visibility so that the function is not deleted by MLIR
            // because unused. Changing it is OK here because the function will
            // be deleted anyway in the second filtering phase.
            internalFunc.setVisibility(mlir::SymbolTable::Visibility::Public);
            continue;
          }
          // If the callOp has users then replace them with Undef values.
          if (!callOp->use_empty()) {
            SmallVector<Value> undefResults;
            for (Value res : callOp->getResults()) {
              opBuilder.setInsertionPoint(callOp);
              undefResults.emplace_back(
                  fir::UndefOp::create(opBuilder, res.getLoc(), res.getType()));
            }
            callOp->replaceAllUsesWith(undefResults);
          }
          // Remove the callOp
          callOp->erase();
        }
        if (!hasTargetRegion) {
          funcOp.erase();
          return WalkResult::skip();
        }
        if (declareTargetOp)
          declareTargetOp.setDeclareTarget(
              declareType, omp::DeclareTargetCaptureClause::to,
              declareTargetOp.getDeclareTargetAutomap());
      }
      return WalkResult::advance();
    });

    checkDeviceImplementationStatus(op);
  }
};
} // namespace
