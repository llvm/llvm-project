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
#include "flang/Optimizer/OpenMP/Utils.h"

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Frontend/OpenMP/OMPConstants.h"

namespace flangomp {
#define GEN_PASS_DEF_AUTOMAPTOTARGETDATAPASS
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

using namespace mlir;

namespace {
class AutomapToTargetDataPass
    : public flangomp::impl::AutomapToTargetDataPassBase<
          AutomapToTargetDataPass> {
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
      if (flangomp::needsBoundsOps(memOp.getMemref()))
        bounds = flangomp::genBoundsOps(builder, memOp.getMemref());

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
  }
};
} // namespace
