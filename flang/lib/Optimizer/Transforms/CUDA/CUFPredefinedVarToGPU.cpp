//===-- CUFPredefinedVarToGPU.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace fir {
#define GEN_PASS_DEF_CUFPREDEFINEDVARTOGPU
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace mlir;

namespace {

static constexpr llvm::StringRef builtinsModuleName = "__fortran_builtins";
static constexpr llvm::StringRef builtinVarPrefix = "__builtin_";
static constexpr llvm::StringRef threadidx = "threadidx";
static constexpr llvm::StringRef blockidx = "blockidx";
static constexpr llvm::StringRef blockdim = "blockdim";
static constexpr llvm::StringRef griddim = "griddim";

static constexpr unsigned field_x = 0;
static constexpr unsigned field_y = 1;
static constexpr unsigned field_z = 2;

std::string mangleBuiltin(llvm::StringRef varName) {
  return "_QM" + builtinsModuleName.str() + "E" + builtinVarPrefix.str() +
         varName.str();
}

template <typename OpTy>
static void
processCoordinateOp(mlir::OpBuilder &builder, fir::CoordinateOp coordOp,
                    unsigned fieldIdx, bool incrementByOne,
                    llvm::SmallVectorImpl<mlir::Operation *> &opsToDelete) {
  std::optional<llvm::ArrayRef<int32_t>> fieldIndices =
      coordOp.getFieldIndices();
  assert(fieldIndices && fieldIndices->size() == 1 &&
         "expect only one coordinate");
  if (static_cast<unsigned>((*fieldIndices)[0]) == fieldIdx) {
    for (mlir::OpOperand &coordUse : coordOp.getResult().getUses()) {
      assert(mlir::isa<fir::LoadOp>(coordUse.getOwner()) &&
             "only expect load op");
      auto loadOp = mlir::dyn_cast<fir::LoadOp>(coordUse.getOwner());
      // Use the loadOp's loc as the location info for the register read op.
      mlir::Location loc = loadOp.getLoc();
      builder.setInsertionPoint(loadOp);
      mlir::Value gpuValue = OpTy::create(builder, loc, builder.getI32Type());
      if (incrementByOne) {
        auto c1 = mlir::arith::ConstantOp::create(
            builder, loc, builder.getI32Type(), builder.getI32IntegerAttr(1));
        gpuValue = mlir::arith::AddIOp::create(builder, loc, gpuValue, c1);
      }
      loadOp.getResult().replaceAllUsesWith(gpuValue);
      opsToDelete.push_back(loadOp);
    }
  }
}

template <typename OpTyX, typename OpTyY, typename OpTyZ>
static void
processDeclareOp(mlir::OpBuilder &builder, fir::DeclareOp declareOp,
                 llvm::StringRef builtinVar, bool incrementByOne,
                 llvm::SmallVectorImpl<mlir::Operation *> &opsToDelete,
                 llvm::SmallPtrSetImpl<mlir::Operation *> &memrefDefiningOps) {
  if (declareOp.getUniqName().str().compare(builtinVar) == 0) {
    for (mlir::OpOperand &use : declareOp.getResult().getUses()) {
      fir::CoordinateOp coordOp =
          mlir::dyn_cast<fir::CoordinateOp>(use.getOwner());
      processCoordinateOp<OpTyX>(builder, coordOp, field_x, incrementByOne,
                                 opsToDelete);
      processCoordinateOp<OpTyY>(builder, coordOp, field_y, incrementByOne,
                                 opsToDelete);
      processCoordinateOp<OpTyZ>(builder, coordOp, field_z, incrementByOne,
                                 opsToDelete);
      opsToDelete.push_back(coordOp);
    }
    opsToDelete.push_back(declareOp.getOperation());
    // The backing fir.address_of may be shared by several declares (e.g. after
    // CSE coalesces them when a device routine is inlined into a kernel).
    // Collect it de-duplicated and erase it only once all declares are gone.
    if (mlir::Operation *memrefOp = declareOp.getMemref().getDefiningOp())
      memrefDefiningOps.insert(memrefOp);
  }
}

struct CUFPredefinedVarToGPU
    : public fir::impl::CUFPredefinedVarToGPUBase<CUFPredefinedVarToGPU> {

  void rewritePredefinedVars(mlir::Region &region) {
    if (region.empty())
      return;

    bool hasPredefinedDeclares = false;
    region.walk([&](fir::DeclareOp declareOp) {
      llvm::StringRef uniqName = declareOp.getUniqName();
      hasPredefinedDeclares |= uniqName == mangleBuiltin(threadidx) ||
                               uniqName == mangleBuiltin(blockidx) ||
                               uniqName == mangleBuiltin(blockdim) ||
                               uniqName == mangleBuiltin(griddim);
    });
    if (!hasPredefinedDeclares)
      return;

    mlir::OpBuilder builder(region.getContext());
    llvm::SmallVector<mlir::Operation *> opsToDelete;
    llvm::SmallPtrSet<mlir::Operation *, 4> memrefDefiningOps;
    region.walk([&](fir::DeclareOp declareOp) {
      processDeclareOp<mlir::NVVM::ThreadIdXOp, mlir::NVVM::ThreadIdYOp,
                       mlir::NVVM::ThreadIdZOp>(
          builder, declareOp, mangleBuiltin(threadidx),
          /*incrementByOne=*/true, opsToDelete, memrefDefiningOps);
      processDeclareOp<mlir::NVVM::BlockIdXOp, mlir::NVVM::BlockIdYOp,
                       mlir::NVVM::BlockIdZOp>(
          builder, declareOp, mangleBuiltin(blockidx),
          /*incrementByOne=*/true, opsToDelete, memrefDefiningOps);
      processDeclareOp<mlir::NVVM::BlockDimXOp, mlir::NVVM::BlockDimYOp,
                       mlir::NVVM::BlockDimZOp>(
          builder, declareOp, mangleBuiltin(blockdim),
          /*incrementByOne=*/false, opsToDelete, memrefDefiningOps);
      processDeclareOp<mlir::NVVM::GridDimXOp, mlir::NVVM::GridDimYOp,
                       mlir::NVVM::GridDimZOp>(
          builder, declareOp, mangleBuiltin(griddim),
          /*incrementByOne=*/false, opsToDelete, memrefDefiningOps);
    });

    for (auto *op : opsToDelete)
      op->erase();
    // Erase each backing fir.address_of once, and only if no other user (e.g. a
    // non-predefined declare) still references it.
    for (auto *op : memrefDefiningOps)
      if (op->use_empty())
        op->erase();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (funcOp.getBody().empty())
      return;

    bool rewrittenWholeFunction = false;
    if (auto cudaProcAttr =
            funcOp.getOperation()->getAttrOfType<cuf::ProcAttributeAttr>(
                cuf::getProcAttrName())) {
      if (cudaProcAttr.getValue() == cuf::ProcAttribute::Device ||
          cudaProcAttr.getValue() == cuf::ProcAttribute::Global ||
          cudaProcAttr.getValue() == cuf::ProcAttribute::GridGlobal ||
          cudaProcAttr.getValue() == cuf::ProcAttribute::HostDevice) {
        rewritePredefinedVars(funcOp.getRegion());
        rewrittenWholeFunction = true;
      }
    }

    if (rewrittenWholeFunction)
      return;

    // Host functions containing cuf.kernel or OpenACC compute regions can
    // still carry predefined vars in the kernel body. Rewrite them in-place.
    funcOp.walk([&](cuf::KernelOp kernelOp) {
      rewritePredefinedVars(kernelOp.getRegion());
    });
    funcOp.walk([&](mlir::acc::ComputeRegionOpInterface computeOp) {
      rewritePredefinedVars(computeOp->getRegion(0));
    });
  }
};

} // end anonymous namespace
