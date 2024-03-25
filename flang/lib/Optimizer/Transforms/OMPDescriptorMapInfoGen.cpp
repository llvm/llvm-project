//===- OMPDescriptorMapInfoGen.cpp
//---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// \file
/// An OpenMP dialect related pass for FIR/HLFIR which expands MapInfoOp's
/// containing descriptor related types (fir::BoxType's) into multiple
/// MapInfoOp's containing the parent descriptor and pointer member components
/// for individual mapping, treating the descriptor type as a record type for
/// later lowering in the OpenMP dialect.
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <iterator>

namespace fir {
#define GEN_PASS_DEF_OMPDESCRIPTORMAPINFOGENPASS
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

namespace {
class OMPDescriptorMapInfoGenPass
    : public fir::impl::OMPDescriptorMapInfoGenPassBase<
          OMPDescriptorMapInfoGenPass> {

  void genDescriptorMemberMaps(mlir::omp::MapInfoOp op,
                               fir::FirOpBuilder &builder,
                               mlir::Operation *target) {
    mlir::Location loc = builder.getUnknownLoc();
    mlir::Value descriptor = op.getVarPtr();

    // If we enter this function, but the mapped type itself is not the
    // descriptor, then it's likely the address of the descriptor so we
    // must retrieve the descriptor SSA.
    if (!fir::isTypeWithDescriptor(op.getVarType())) {
      if (auto addrOp = mlir::dyn_cast_if_present<fir::BoxAddrOp>(
              op.getVarPtr().getDefiningOp())) {
        descriptor = addrOp.getVal();
      }
    }

    // The fir::BoxOffsetOp only works with !fir.ref<!fir.box<...>> types, as
    // allowing it to access non-reference box operations can cause some
    // problematic SSA IR. However, in the case of assumed shape's the type
    // is not a !fir.ref, in these cases to retrieve the appropriate
    // !fir.ref<!fir.box<...>> to access the data we need to map we must
    // perform an alloca and then store to it and retrieve the data from the new
    // alloca.
    if (mlir::isa<fir::BaseBoxType>(descriptor.getType())) {
      mlir::OpBuilder::InsertPoint insPt = builder.saveInsertionPoint();
      builder.setInsertionPointToStart(builder.getAllocaBlock());
      auto alloca = builder.create<fir::AllocaOp>(loc, descriptor.getType());
      builder.restoreInsertionPoint(insPt);
      builder.create<fir::StoreOp>(loc, descriptor, alloca);
      descriptor = alloca;
    }

    mlir::Value baseAddrAddr = builder.create<fir::BoxOffsetOp>(
        loc, descriptor, fir::BoxFieldAttr::base_addr);

    // Member of the descriptor pointing at the allocated data
    mlir::Value baseAddr = builder.create<mlir::omp::MapInfoOp>(
        loc, baseAddrAddr.getType(), descriptor,
        llvm::cast<mlir::omp::PointerLikeType>(
            fir::unwrapRefType(baseAddrAddr.getType()))
            .getElementType(),
        baseAddrAddr, mlir::SmallVector<mlir::Value>{}, op.getBounds(),
        builder.getIntegerAttr(builder.getIntegerType(64, false),
                               op.getMapType().value()),
        builder.getAttr<mlir::omp::VariableCaptureKindAttr>(
            mlir::omp::VariableCaptureKind::ByRef),
        builder.getStringAttr("") /*name*/);

    // TODO: map the addendum segment of the descriptor, similarly to the
    // above base address/data pointer member.

    if (auto mapClauseOwner =
            llvm::dyn_cast<mlir::omp::MapClauseOwningOpInterface>(target)) {
      llvm::SmallVector<mlir::Value> newMapOps;
      mlir::OperandRange mapOperandsArr = mapClauseOwner.getMapOperands();

      for (size_t i = 0; i < mapOperandsArr.size(); ++i) {
        if (mapOperandsArr[i] == op) {
          // Push new implicit maps generated for the descriptor.
          newMapOps.push_back(baseAddr);

          // for TargetOp's which have IsolatedFromAbove we must align the
          // new additional map operand with an appropriate BlockArgument,
          // as the printing and later processing currently requires a 1:1
          // mapping of BlockArgs to MapInfoOp's at the same placement in
          // each array (BlockArgs and MapOperands).
          if (auto targetOp = llvm::dyn_cast<mlir::omp::TargetOp>(target))
            targetOp.getRegion().insertArgument(i, baseAddr.getType(), loc);
        }
        newMapOps.push_back(mapOperandsArr[i]);
      }
      mapClauseOwner.getMapOperandsMutable().assign(newMapOps);
    }

    mlir::Value newDescParentMapOp = builder.create<mlir::omp::MapInfoOp>(
        op->getLoc(), op.getResult().getType(), descriptor,
        fir::unwrapRefType(descriptor.getType()), mlir::Value{},
        mlir::SmallVector<mlir::Value>{baseAddr},
        mlir::SmallVector<mlir::Value>{},
        builder.getIntegerAttr(builder.getIntegerType(64, false),
                               op.getMapType().value()),
        op.getMapCaptureTypeAttr(), op.getNameAttr());
    op.replaceAllUsesWith(newDescParentMapOp);
    op->erase();
  }

  // This pass executes on mlir::ModuleOp's finding omp::MapInfoOp's containing
  // descriptor based types (allocatables, pointers, assumed shape etc.) and
  // expanding them into multiple omp::MapInfoOp's for each pointer member
  // contained within the descriptor.
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::ModuleOp module = func->getParentOfType<mlir::ModuleOp>();
    fir::KindMapping kindMap = fir::getKindMapping(module);
    fir::FirOpBuilder builder{module, std::move(kindMap)};

    func->walk([&](mlir::omp::MapInfoOp op) {
      if (fir::isTypeWithDescriptor(op.getVarType()) ||
          mlir::isa_and_present<fir::BoxAddrOp>(
              op.getVarPtr().getDefiningOp())) {
        builder.setInsertionPoint(op);
        // TODO: Currently only supports a single user for the MapInfoOp, this
        // is fine for the moment as the Fortran Frontend will generate a
        // new MapInfoOp per Target operation for the moment. However, when/if
        // we optimise/cleanup the IR, it likely isn't too difficult to
        // extend this function, it would require some modification to create a
        // single new MapInfoOp per new MapInfoOp generated and share it across
        // all users appropriately, making sure to only add a single member link
        // per new generation for the original originating descriptor MapInfoOp.
        assert(llvm::hasSingleElement(op->getUsers()) &&
               "OMPDescriptorMapInfoGen currently only supports single users "
               "of a MapInfoOp");
        genDescriptorMemberMaps(op, builder, *op->getUsers().begin());
      }
    });
  }
};

} // namespace

namespace fir {
std::unique_ptr<mlir::Pass> createOMPDescriptorMapInfoGenPass() {
  return std::make_unique<OMPDescriptorMapInfoGenPass>();
}
} // namespace fir
