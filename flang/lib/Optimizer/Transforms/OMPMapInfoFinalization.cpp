//===- OMPMapInfoFinalization.cpp
//---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// \file
/// An OpenMP dialect related pass for FIR/HLFIR which performs some
/// pre-processing of MapInfoOp's after the module has been lowered to
/// finalize them.
///
/// For example, it expands MapInfoOp's containing descriptor related
/// types (fir::BoxType's) into multiple MapInfoOp's containing the parent
/// descriptor and pointer member components for individual mapping,
/// treating the descriptor type as a record type for later lowering in the
/// OpenMP dialect.
///
/// The pass also adds MapInfoOp's that are members of a parent object but are
/// not directly used in the body of a target region to its BlockArgument list
/// to maintain consistency across all MapInfoOp's tied to a region directly or
/// indirectly via a parent object.
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
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include <iterator>

namespace fir {
#define GEN_PASS_DEF_OMPMAPINFOFINALIZATIONPASS
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

namespace {
class OMPMapInfoFinalizationPass
    : public fir::impl::OMPMapInfoFinalizationPassBase<
          OMPMapInfoFinalizationPass> {

  void genDescriptorMemberMaps(mlir::omp::MapInfoOp op,
                               fir::FirOpBuilder &builder,
                               mlir::Operation *target) {
    mlir::Location loc = op.getLoc();
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
      mlir::Block *allocaBlock = builder.getAllocaBlock();
      assert(allocaBlock && "No alloca block found for this top level op");
      builder.setInsertionPointToStart(allocaBlock);
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
        mlir::TypeAttr::get(llvm::cast<mlir::omp::PointerLikeType>(
                                fir::unwrapRefType(baseAddrAddr.getType()))
                                .getElementType()),
        baseAddrAddr, /*members=*/mlir::SmallVector<mlir::Value>{},
        /*member_index=*/mlir::DenseIntElementsAttr{}, op.getBounds(),
        builder.getIntegerAttr(builder.getIntegerType(64, false),
                               op.getMapType().value()),
        builder.getAttr<mlir::omp::VariableCaptureKindAttr>(
            mlir::omp::VariableCaptureKind::ByRef),
        /*name=*/builder.getStringAttr(""),
        /*partial_map=*/builder.getBoolAttr(false));

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
        mlir::TypeAttr::get(fir::unwrapRefType(descriptor.getType())),
        /*varPtrPtr=*/mlir::Value{},
        /*members=*/mlir::SmallVector<mlir::Value>{baseAddr},
        /*members_index=*/
        mlir::DenseIntElementsAttr::get(
            mlir::VectorType::get(
                llvm::ArrayRef<int64_t>({1, 1}),
                mlir::IntegerType::get(builder.getContext(), 32)),
            llvm::ArrayRef<int32_t>({0})),
        /*bounds=*/mlir::SmallVector<mlir::Value>{},
        builder.getIntegerAttr(builder.getIntegerType(64, false),
                               op.getMapType().value()),
        op.getMapCaptureTypeAttr(), op.getNameAttr(), op.getPartialMapAttr());
    op.replaceAllUsesWith(newDescParentMapOp);
    op->erase();
  }

  // We add all mapped record members not directly used in the target region
  // to the block arguments in front of their parent and we place them into
  // the map operands list for consistency.
  //
  // These indirect uses (via accesses to their parent) will still be
  // mapped individually in most cases, and a parent mapping doesn't
  // guarantee the parent will be mapped in its totality, partial
  // mapping is common.
  //
  // For example:
  //    map(tofrom: x%y)
  //
  // Will generate a mapping for "x" (the parent) and "y" (the member).
  // The parent "x" will not be mapped, but the member "y" will.
  // However, we must have the parent as a BlockArg and MapOperand
  // in these cases, to maintain the correct uses within the region and
  // to help tracking that the member is part of a larger object.
  //
  // In the case of:
  //    map(tofrom: x%y, x%z)
  //
  // The parent member becomes more critical, as we perform a partial
  // structure mapping where we link the mapping of the members y
  // and z together via the parent x. We do this at a kernel argument
  // level in LLVM IR and not just MLIR, which is important to maintain
  // similarity to Clang and for the runtime to do the correct thing.
  // However, we still do not map the structure in its totality but
  // rather we generate an un-sized "binding" map entry for it.
  //
  // In the case of:
  //    map(tofrom: x, x%y, x%z)
  //
  // We do actually map the entirety of "x", so the explicit mapping of
  // x%y, x%z becomes unnecessary. It is redundant to write this from a
  // Fortran OpenMP perspective (although it is legal), as even if the
  // members were allocatables or pointers, we are mandated by the
  // specification to map these (and any recursive components) in their
  // entirety, which is different to the C++ equivalent, which requires
  // explicit mapping of these segments.
  void addImplicitMembersToTarget(mlir::omp::MapInfoOp op,
                                  fir::FirOpBuilder &builder,
                                  mlir::Operation *target) {
    auto mapClauseOwner =
        llvm::dyn_cast<mlir::omp::MapClauseOwningOpInterface>(target);
    if (!mapClauseOwner)
      return;

    llvm::SmallVector<mlir::Value> newMapOps;
    mlir::OperandRange mapOperandsArr = mapClauseOwner.getMapOperands();
    auto targetOp = llvm::dyn_cast<mlir::omp::TargetOp>(target);

    for (size_t i = 0; i < mapOperandsArr.size(); ++i) {
      if (mapOperandsArr[i] == op) {
        for (auto [j, mapMember] : llvm::enumerate(op.getMembers())) {
          newMapOps.push_back(mapMember);
          // for TargetOp's which have IsolatedFromAbove we must align the
          // new additional map operand with an appropriate BlockArgument,
          // as the printing and later processing currently requires a 1:1
          // mapping of BlockArgs to MapInfoOp's at the same placement in
          // each array (BlockArgs and MapOperands).
          if (targetOp) {
            targetOp.getRegion().insertArgument(i + j, mapMember.getType(),
                                                targetOp->getLoc());
          }
        }
      }
      newMapOps.push_back(mapOperandsArr[i]);
    }
    mapClauseOwner.getMapOperandsMutable().assign(newMapOps);
  }

  // This pass executes on omp::MapInfoOp's containing descriptor based types
  // (allocatables, pointers, assumed shape etc.) and expanding them into
  // multiple omp::MapInfoOp's for each pointer member contained within the
  // descriptor.
  //
  // From the perspective of the MLIR pass manager this runs on the top level
  // operation (usually function) containing the MapInfoOp because this pass
  // will mutate siblings of MapInfoOp.
  void runOnOperation() override {
    mlir::ModuleOp module =
        mlir::dyn_cast_or_null<mlir::ModuleOp>(getOperation());
    if (!module)
      module = getOperation()->getParentOfType<mlir::ModuleOp>();
    fir::KindMapping kindMap = fir::getKindMapping(module);
    fir::FirOpBuilder builder{module, std::move(kindMap)};

    getOperation()->walk([&](mlir::omp::MapInfoOp op) {
      // TODO: Currently only supports a single user for the MapInfoOp, this
      // is fine for the moment as the Fortran Frontend will generate a
      // new MapInfoOp per Target operation for the moment. However, when/if
      // we optimise/cleanup the IR, it likely isn't too difficult to
      // extend this function, it would require some modification to create a
      // single new MapInfoOp per new MapInfoOp generated and share it across
      // all users appropriately, making sure to only add a single member link
      // per new generation for the original originating descriptor MapInfoOp.
      assert(llvm::hasSingleElement(op->getUsers()) &&
             "OMPMapInfoFinalization currently only supports single users "
             "of a MapInfoOp");

      if (!op.getMembers().empty()) {
        addImplicitMembersToTarget(op, builder, *op->getUsers().begin());
      } else if (fir::isTypeWithDescriptor(op.getVarType()) ||
                 mlir::isa_and_present<fir::BoxAddrOp>(
                     op.getVarPtr().getDefiningOp())) {
        builder.setInsertionPoint(op);
        genDescriptorMemberMaps(op, builder, *op->getUsers().begin());
      }
    });
  }
};

} // namespace
