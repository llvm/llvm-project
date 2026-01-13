//===- MarkDeclareTarget.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Mark functions called from explicit target code as implicitly declare target.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/OpenMP/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

namespace flangomp {
#define GEN_PASS_DEF_MARKDECLARETARGETPASS
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

namespace {
class MarkDeclareTargetPass
    : public flangomp::impl::MarkDeclareTargetPassBase<MarkDeclareTargetPass> {

  struct ParentInfo {
    mlir::omp::DeclareTargetDeviceType devTy;
    mlir::omp::DeclareTargetCaptureClause capClause;
    bool automap;
  };

  void processSymbolRef(mlir::SymbolRefAttr symRef, ParentInfo parentInfo,
                        llvm::SmallPtrSet<mlir::Operation *, 16> visited) {
    if (auto currFOp =
            getOperation().lookupSymbol<mlir::func::FuncOp>(symRef)) {
      auto current = llvm::dyn_cast<mlir::omp::DeclareTargetInterface>(
          currFOp.getOperation());

      if (current.isDeclareTarget()) {
        auto currentDt = current.getDeclareTargetDeviceType();

        // Found the same function twice, with different device_types,
        // mark as Any as it belongs to both
        if (currentDt != parentInfo.devTy &&
            currentDt != mlir::omp::DeclareTargetDeviceType::any) {
          current.setDeclareTarget(mlir::omp::DeclareTargetDeviceType::any,
                                   current.getDeclareTargetCaptureClause(),
                                   current.getDeclareTargetAutomap());
        }
      } else {
        current.setDeclareTarget(parentInfo.devTy, parentInfo.capClause,
                                 parentInfo.automap);
      }

      markNestedFuncs(parentInfo, currFOp, visited);
    }
  }

  void processReductionRefs(std::optional<mlir::ArrayAttr> symRefs,
                            ParentInfo parentInfo,
                            llvm::SmallPtrSet<mlir::Operation *, 16> visited) {
    if (!symRefs)
      return;

    for (auto symRef : symRefs->getAsRange<mlir::SymbolRefAttr>()) {
      if (auto declareReductionOp =
              getOperation().lookupSymbol<mlir::omp::DeclareReductionOp>(
                  symRef)) {
        markNestedFuncs(parentInfo, declareReductionOp, visited);
      }
    }
  }

  void
  processReductionClauses(mlir::Operation *op, ParentInfo parentInfo,
                          llvm::SmallPtrSet<mlir::Operation *, 16> visited) {
    llvm::TypeSwitch<mlir::Operation &>(*op)
        .Case([&](mlir::omp::LoopOp op) {
          processReductionRefs(op.getReductionSyms(), parentInfo, visited);
        })
        .Case([&](mlir::omp::ParallelOp op) {
          processReductionRefs(op.getReductionSyms(), parentInfo, visited);
        })
        .Case([&](mlir::omp::SectionsOp op) {
          processReductionRefs(op.getReductionSyms(), parentInfo, visited);
        })
        .Case([&](mlir::omp::SimdOp op) {
          processReductionRefs(op.getReductionSyms(), parentInfo, visited);
        })
        .Case([&](mlir::omp::TargetOp op) {
          processReductionRefs(op.getInReductionSyms(), parentInfo, visited);
        })
        .Case([&](mlir::omp::TaskgroupOp op) {
          processReductionRefs(op.getTaskReductionSyms(), parentInfo, visited);
        })
        .Case([&](mlir::omp::TaskloopOp op) {
          processReductionRefs(op.getReductionSyms(), parentInfo, visited);
          processReductionRefs(op.getInReductionSyms(), parentInfo, visited);
        })
        .Case([&](mlir::omp::TaskOp op) {
          processReductionRefs(op.getInReductionSyms(), parentInfo, visited);
        })
        .Case([&](mlir::omp::TeamsOp op) {
          processReductionRefs(op.getReductionSyms(), parentInfo, visited);
        })
        .Case([&](mlir::omp::WsloopOp op) {
          processReductionRefs(op.getReductionSyms(), parentInfo, visited);
        })
        .Default([](mlir::Operation &) {});
  }

  void markNestedFuncs(ParentInfo parentInfo, mlir::Operation *currOp,
                       llvm::SmallPtrSet<mlir::Operation *, 16> visited) {
    if (visited.contains(currOp))
      return;
    visited.insert(currOp);

    currOp->walk([&, this](mlir::Operation *op) {
      if (auto callOp = llvm::dyn_cast<mlir::CallOpInterface>(op)) {
        if (auto symRef = llvm::dyn_cast_if_present<mlir::SymbolRefAttr>(
                callOp.getCallableForCallee())) {
          processSymbolRef(symRef, parentInfo, visited);
        }
      }
      processReductionClauses(op, parentInfo, visited);
    });
  }

  // This pass executes on mlir::ModuleOp's marking functions contained within
  // as implicitly declare target if they are called from within an explicitly
  // marked declare target function or a target region (TargetOp)
  void runOnOperation() override {
    for (auto functionOp : getOperation().getOps<mlir::func::FuncOp>()) {
      auto declareTargetOp = llvm::dyn_cast<mlir::omp::DeclareTargetInterface>(
          functionOp.getOperation());
      if (declareTargetOp.isDeclareTarget()) {
        llvm::SmallPtrSet<mlir::Operation *, 16> visited;
        ParentInfo parentInfo{declareTargetOp.getDeclareTargetDeviceType(),
                              declareTargetOp.getDeclareTargetCaptureClause(),
                              declareTargetOp.getDeclareTargetAutomap()};
        markNestedFuncs(parentInfo, functionOp, visited);
      }
    }

    // TODO: Extend to work with reverse-offloading, this shouldn't
    // require too much effort, just need to check the device clause
    // when it's lowering has been implemented and change the
    // DeclareTargetDeviceType argument from nohost to host depending on
    // the contents of the device clause
    getOperation()->walk([&](mlir::omp::TargetOp tarOp) {
      llvm::SmallPtrSet<mlir::Operation *, 16> visited;
      ParentInfo parentInfo = {
          /*devTy=*/mlir::omp::DeclareTargetDeviceType::nohost,
          /*capClause=*/mlir::omp::DeclareTargetCaptureClause::to,
          /*automap=*/false,
      };
      markNestedFuncs(parentInfo, tarOp, visited);
    });
  }
};
} // namespace
