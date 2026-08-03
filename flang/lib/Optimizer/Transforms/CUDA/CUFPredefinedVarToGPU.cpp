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

template <typename OpTyX, typename OpTyY, typename OpTyZ>
static void createForAllDimensions(mlir::OpBuilder &builder, mlir::Location loc,
                                   mlir::Value c1,
                                   SmallVectorImpl<mlir::Value> &values,
                                   bool incrementByOne = false) {
  if (incrementByOne) {
    auto baseX = OpTyX::create(builder, loc, builder.getI32Type());
    values.push_back(mlir::arith::AddIOp::create(builder, loc, baseX, c1));
    auto baseY = OpTyY::create(builder, loc, builder.getI32Type());
    values.push_back(mlir::arith::AddIOp::create(builder, loc, baseY, c1));
    auto baseZ = OpTyZ::create(builder, loc, builder.getI32Type());
    values.push_back(mlir::arith::AddIOp::create(builder, loc, baseZ, c1));
  } else {
    values.push_back(OpTyX::create(builder, loc, builder.getI32Type()));
    values.push_back(OpTyY::create(builder, loc, builder.getI32Type()));
    values.push_back(OpTyZ::create(builder, loc, builder.getI32Type()));
  }
}

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

static void
processCoordinateOp(mlir::OpBuilder &builder, mlir::Location loc,
                    fir::CoordinateOp coordOp, unsigned fieldIdx,
                    mlir::Value &gpuValue,
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
      loadOp.getResult().replaceAllUsesWith(gpuValue);
      opsToDelete.push_back(loadOp);
    }
  }
}

static void
processDeclareOp(mlir::OpBuilder &builder, mlir::Location loc,
                 fir::DeclareOp declareOp, llvm::StringRef builtinVar,
                 llvm::SmallVectorImpl<mlir::Value> &gpuValues,
                 llvm::SmallVectorImpl<mlir::Operation *> &opsToDelete,
                 llvm::SmallPtrSetImpl<mlir::Operation *> &memrefDefiningOps) {
  if (declareOp.getUniqName().str().compare(builtinVar) == 0) {
    for (mlir::OpOperand &use : declareOp.getResult().getUses()) {
      fir::CoordinateOp coordOp =
          mlir::dyn_cast<fir::CoordinateOp>(use.getOwner());
      processCoordinateOp(builder, loc, coordOp, field_x, gpuValues[0],
                          opsToDelete);
      processCoordinateOp(builder, loc, coordOp, field_y, gpuValues[1],
                          opsToDelete);
      processCoordinateOp(builder, loc, coordOp, field_z, gpuValues[2],
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

  void rewritePredefinedVars(mlir::Region &region, mlir::Location loc) {
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
    builder.setInsertionPointToStart(&region.front());
    auto c1 = mlir::arith::ConstantOp::create(
        builder, loc, builder.getI32Type(), builder.getI32IntegerAttr(1));
    llvm::SmallVector<mlir::Value, 3> threadids, blockids, blockdims, griddims;
    createForAllDimensions<mlir::NVVM::ThreadIdXOp, mlir::NVVM::ThreadIdYOp,
                           mlir::NVVM::ThreadIdZOp>(builder, loc, c1, threadids,
                                                    /*incrementByOne=*/true);
    createForAllDimensions<mlir::NVVM::BlockIdXOp, mlir::NVVM::BlockIdYOp,
                           mlir::NVVM::BlockIdZOp>(builder, loc, c1, blockids,
                                                   /*incrementByOne=*/true);
    createForAllDimensions<mlir::NVVM::GridDimXOp, mlir::NVVM::GridDimYOp,
                           mlir::NVVM::GridDimZOp>(builder, loc, c1, griddims);
    createForAllDimensions<mlir::NVVM::BlockDimXOp, mlir::NVVM::BlockDimYOp,
                           mlir::NVVM::BlockDimZOp>(builder, loc, c1,
                                                    blockdims);

    llvm::SmallVector<mlir::Operation *> opsToDelete;
    llvm::SmallPtrSet<mlir::Operation *, 4> memrefDefiningOps;
    region.walk([&](fir::DeclareOp declareOp) {
      processDeclareOp(builder, loc, declareOp, mangleBuiltin(threadidx),
                       threadids, opsToDelete, memrefDefiningOps);
      processDeclareOp(builder, loc, declareOp, mangleBuiltin(blockidx),
                       blockids, opsToDelete, memrefDefiningOps);
      processDeclareOp(builder, loc, declareOp, mangleBuiltin(blockdim),
                       blockdims, opsToDelete, memrefDefiningOps);
      processDeclareOp(builder, loc, declareOp, mangleBuiltin(griddim),
                       griddims, opsToDelete, memrefDefiningOps);
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
        rewritePredefinedVars(funcOp.getRegion(), funcOp.getLoc());
        rewrittenWholeFunction = true;
      }
    }

    if (rewrittenWholeFunction)
      return;

    // Host functions containing cuf.kernel or OpenACC compute regions can
    // still carry predefined vars in the kernel body. Rewrite them in-place.
    funcOp.walk([&](cuf::KernelOp kernelOp) {
      rewritePredefinedVars(kernelOp.getRegion(), kernelOp.getLoc());
    });
    funcOp.walk([&](mlir::acc::ComputeRegionOpInterface computeOp) {
      rewritePredefinedVars(computeOp->getRegion(0), computeOp->getLoc());
    });
  }
};

} // end anonymous namespace
