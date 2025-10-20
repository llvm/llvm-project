//===- AutomapToTargetData.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/DirectivesCommon.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include <algorithm>

namespace flangomp {
#define GEN_PASS_DEF_AUTOMAPTOTARGETDATAPASS
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

using namespace mlir;

namespace {
class AutomapToTargetDataPass
    : public flangomp::impl::AutomapToTargetDataPassBase<
          AutomapToTargetDataPass> {

  // Returns true if the variable has a dynamic size and therefore requires
  // bounds operations to describe its extents.
  inline bool needsBoundsOps(mlir::Value var) {
    assert(mlir::isa<mlir::omp::PointerLikeType>(var.getType()) &&
           "only pointer like types expected");
    mlir::Type t = fir::unwrapRefType(var.getType());
    if (mlir::Type inner = fir::dyn_cast_ptrOrBoxEleTy(t))
      return fir::hasDynamicSize(inner);
    return fir::hasDynamicSize(t);
  }

  // Generate MapBoundsOp operations for the variable if required.
  inline void genBoundsOps(fir::FirOpBuilder &builder, mlir::Value var,
                           llvm::SmallVectorImpl<mlir::Value> &boundsOps) {
    mlir::Location loc = var.getLoc();
    fir::factory::AddrAndBoundsInfo info =
        fir::factory::getDataOperandBaseAddr(builder, var,
                                             /*isOptional=*/false, loc);
    fir::ExtendedValue exv =
        hlfir::translateToExtendedValue(loc, builder, hlfir::Entity{info.addr},
                                        /*contiguousHint=*/true)
            .first;
    llvm::SmallVector<mlir::Value> tmp =
        fir::factory::genImplicitBoundsOps<mlir::omp::MapBoundsOp,
                                           mlir::omp::MapBoundsType>(
            builder, info, exv, /*dataExvIsAssumedSize=*/false, loc);
    llvm::append_range(boundsOps, tmp);
  }

  void findRelatedAllocmemFreemem(fir::AddrOfOp addressOfOp,
                                  llvm::DenseSet<fir::StoreOp> &allocmems,
                                  llvm::DenseSet<fir::LoadOp> &freemems) {
    assert(addressOfOp->hasOneUse() && "op must have single use");

    auto declaredRef =
        cast<hlfir::DeclareOp>(*addressOfOp->getUsers().begin())->getResult(0);

    for (Operation *refUser : declaredRef.getUsers()) {
      if (auto storeOp = dyn_cast<fir::StoreOp>(refUser))
        if (auto emboxOp = storeOp.getValue().getDefiningOp<fir::EmboxOp>())
          if (auto allocmemOp =
                  emboxOp.getOperand(0).getDefiningOp<fir::AllocMemOp>())
            allocmems.insert(storeOp);

      if (auto loadOp = dyn_cast<fir::LoadOp>(refUser))
        for (Operation *loadUser : loadOp.getResult().getUsers())
          if (auto boxAddrOp = dyn_cast<fir::BoxAddrOp>(loadUser))
            for (Operation *boxAddrUser : boxAddrOp.getResult().getUsers())
              if (auto freememOp = dyn_cast<fir::FreeMemOp>(boxAddrUser))
                freemems.insert(loadOp);
    }
  }

  void runOnOperation() override {
    ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
    if (!module)
      module = dyn_cast<ModuleOp>(getOperation());
    if (!module)
      return;

    // Build FIR builder for helper utilities.
    fir::KindMapping kindMap = fir::getKindMapping(module);
    fir::FirOpBuilder builder{module, std::move(kindMap)};

    // Collect global variables with AUTOMAP flag.
    llvm::DenseSet<fir::GlobalOp> automapGlobals;
    module.walk([&](fir::GlobalOp globalOp) {
      if (auto iface =
              dyn_cast<omp::DeclareTargetInterface>(globalOp.getOperation()))
        if (iface.isDeclareTarget() && iface.getDeclareTargetAutomap() &&
            iface.getDeclareTargetDeviceType() !=
                omp::DeclareTargetDeviceType::host)
          automapGlobals.insert(globalOp);
    });

    auto addMapInfo = [&](auto globalOp, auto memOp) {
      builder.setInsertionPointAfter(memOp);
      SmallVector<Value> bounds;
      if (needsBoundsOps(memOp.getMemref()))
        genBoundsOps(builder, memOp.getMemref(), bounds);

      omp::TargetEnterExitUpdateDataOperands clauses;
      mlir::omp::MapInfoOp mapInfo = mlir::omp::MapInfoOp::create(
          builder, memOp.getLoc(), memOp.getMemref().getType(),
          memOp.getMemref(),
          TypeAttr::get(fir::unwrapRefType(memOp.getMemref().getType())),
          builder.getIntegerAttr(
              builder.getIntegerType(64, false),
              static_cast<unsigned>(
                  isa<fir::StoreOp>(memOp)
                      ? llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO
                      : llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_DELETE)),
          builder.getAttr<omp::VariableCaptureKindAttr>(
              omp::VariableCaptureKind::ByCopy),
          /*var_ptr_ptr=*/mlir::Value{},
          /*members=*/SmallVector<Value>{},
          /*members_index=*/ArrayAttr{}, bounds,
          /*mapperId=*/mlir::FlatSymbolRefAttr(), globalOp.getSymNameAttr(),
          builder.getBoolAttr(false));
      clauses.mapVars.push_back(mapInfo);
      isa<fir::StoreOp>(memOp)
          ? builder.create<omp::TargetEnterDataOp>(memOp.getLoc(), clauses)
          : builder.create<omp::TargetExitDataOp>(memOp.getLoc(), clauses);
    };

    for (fir::GlobalOp globalOp : automapGlobals) {
      if (auto uses = globalOp.getSymbolUses(module.getOperation())) {
        llvm::DenseSet<fir::StoreOp> allocmemStores;
        llvm::DenseSet<fir::LoadOp> freememLoads;
        for (auto &x : *uses)
          if (auto addrOp = dyn_cast<fir::AddrOfOp>(x.getUser()))
            findRelatedAllocmemFreemem(addrOp, allocmemStores, freememLoads);

        for (auto storeOp : allocmemStores)
          addMapInfo(globalOp, storeOp);

        for (auto loadOp : freememLoads)
          addMapInfo(globalOp, loadOp);
      }
    }

    // Move automapped descriptors from map() to has_device_addr in target ops.
    auto originatesFromAutomapGlobal = [&](mlir::Value varPtr) -> bool {
      if (auto decl = mlir::dyn_cast_or_null<hlfir::DeclareOp>(
              varPtr.getDefiningOp())) {
        if (auto addrOp = mlir::dyn_cast_or_null<fir::AddrOfOp>(
                decl.getMemref().getDefiningOp())) {
          if (auto g =
                  mlir::SymbolTable::lookupNearestSymbolFrom<fir::GlobalOp>(
                      decl, addrOp.getSymbol()))
            return automapGlobals.contains(g);
        }
      }
      return false;
    };

    module.walk([&](mlir::omp::TargetOp target) {
      // Collect candidates to move: descriptor maps of automapped globals.
      llvm::SmallVector<mlir::Value> newMapOps;
      llvm::SmallVector<unsigned> removedIndices;
      llvm::SmallVector<mlir::Value> movedToHDA;
      llvm::SmallVector<mlir::BlockArgument> oldMapArgsForMoved;

      auto mapRange = target.getMapVars();
      newMapOps.reserve(mapRange.size());

      auto argIface = llvm::dyn_cast<mlir::omp::BlockArgOpenMPOpInterface>(
          target.getOperation());
      llvm::ArrayRef<mlir::BlockArgument> mapBlockArgs =
          argIface.getMapBlockArgs();

      for (auto [idx, mapVal] : llvm::enumerate(mapRange)) {
        auto mapOp =
            mlir::dyn_cast<mlir::omp::MapInfoOp>(mapVal.getDefiningOp());
        if (!mapOp) {
          newMapOps.push_back(mapVal);
          continue;
        }

        mlir::Type varTy = fir::unwrapRefType(mapOp.getVarType());
        bool isDescriptor = mlir::isa<fir::BaseBoxType>(varTy);
        if (isDescriptor && originatesFromAutomapGlobal(mapOp.getVarPtr())) {
          movedToHDA.push_back(mapVal);
          removedIndices.push_back(idx);
          oldMapArgsForMoved.push_back(mapBlockArgs[idx]);
        } else {
          newMapOps.push_back(mapVal);
        }
      }

      if (movedToHDA.empty())
        return;

      // Update map vars to exclude moved entries.
      mlir::MutableOperandRange mapMutable = target.getMapVarsMutable();
      mapMutable.assign(newMapOps);

      // Append moved entries to has_device_addr and insert corresponding block
      // arguments.
      mlir::MutableOperandRange hdaMutable =
          target.getHasDeviceAddrVarsMutable();
      llvm::SmallVector<mlir::Value> newHDA;
      newHDA.reserve(hdaMutable.size() + movedToHDA.size());
      llvm::for_each(hdaMutable.getAsOperandRange(),
                     [&](mlir::Value v) { newHDA.push_back(v); });

      unsigned hdaStart = argIface.getHasDeviceAddrBlockArgsStart();
      unsigned oldHdaCount = argIface.numHasDeviceAddrBlockArgs();
      llvm::SmallVector<mlir::BlockArgument> newHDAArgsForMoved;
      unsigned insertIndex = hdaStart + oldHdaCount;
      for (mlir::Value v : movedToHDA) {
        newHDA.push_back(v);
        target->getRegion(0).insertArgument(insertIndex, v.getType(),
                                            v.getLoc());
        // Capture the newly inserted block argument.
        newHDAArgsForMoved.push_back(
            target->getRegion(0).getArgument(insertIndex));
        insertIndex++;
      }
      hdaMutable.assign(newHDA);

      // Redirect uses in the region: replace old map block args with the
      // corresponding new has_device_addr block args.
      for (auto [oldArg, newArg] :
           llvm::zip_equal(oldMapArgsForMoved, newHDAArgsForMoved))
        oldArg.replaceAllUsesWith(newArg);

      // Finally, erase corresponding map block arguments in descending order.
      // Descending order is necessary to avoid index invalidation: erasing
      // arguments from highest to lowest index ensures that earlier erases do
      // not shift the indices of arguments yet to be erased.
      unsigned mapStart = argIface.getMapBlockArgsStart();
      // Convert indices to absolute argument numbers before erasing.
      llvm::SmallVector<unsigned> absArgNos;
      absArgNos.reserve(removedIndices.size());
      for (unsigned idx : removedIndices)
        absArgNos.push_back(mapStart + idx);
      std::sort(absArgNos.begin(), absArgNos.end(), std::greater<>());
      for (unsigned absNo : absArgNos)
        target->getRegion(0).eraseArgument(absNo);
    });
  }
};
} // namespace
