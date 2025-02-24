//===- MapsForPrivatizedSymbols.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// \file
/// An OpenMP dialect related pass for FIR/HLFIR which creates MapInfoOp
/// instances for certain privatized symbols.
/// For example, if an allocatable variable is used in a private clause attached
/// to a omp.target op, then the allocatable variable's descriptor will be
/// needed on the device (e.g. GPU). This descriptor needs to be separately
/// mapped onto the device. This pass creates the necessary omp.map.info ops for
/// this.
//===----------------------------------------------------------------------===//
// TODO:
// 1. Before adding omp.map.info, check if we already have an omp.map.info for
// the variable in question.
// 2. Generalize this for more than just omp.target ops.
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/OpenMP/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Support/Debug.h"
#include <type_traits>

#define DEBUG_TYPE "omp-maps-for-privatized-symbols"

namespace flangomp {
#define GEN_PASS_DEF_MAPSFORPRIVATIZEDSYMBOLSPASS
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp
using namespace mlir;
namespace {
class MapsForPrivatizedSymbolsPass
    : public flangomp::impl::MapsForPrivatizedSymbolsPassBase<
          MapsForPrivatizedSymbolsPass> {

  omp::MapInfoOp createMapInfo(Location loc, Value var,
                               fir::FirOpBuilder &builder) {
    uint64_t mapTypeTo = static_cast<
        std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
        llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO);
    Operation *definingOp = var.getDefiningOp();

    Value varPtr = var;
    // We want the first result of the hlfir.declare op because our goal
    // is to map the descriptor (fir.box or fir.boxchar) and the first
    // result for hlfir.declare is the descriptor if a the symbol being
    // decalred needs a descriptor.
    // Some types are boxed immediately before privatization. These have other
    // operations in between the privatization and the declaration. It is safe
    // to use var directly here because they will be boxed anyway.
    if (auto declOp = llvm::dyn_cast_if_present<hlfir::DeclareOp>(definingOp))
      varPtr = declOp.getBase();

    // If we do not have a reference to descritor, but the descriptor itself
    // then we need to store that on the stack so that we can map the
    // address of the descriptor.
    if (mlir::isa<fir::BaseBoxType>(varPtr.getType()) ||
        mlir::isa<fir::BoxCharType>(varPtr.getType())) {
      OpBuilder::InsertPoint savedInsPoint = builder.saveInsertionPoint();
      mlir::Block *allocaBlock = builder.getAllocaBlock();
      assert(allocaBlock && "No allocablock  found for a funcOp");
      builder.setInsertionPointToStart(allocaBlock);
      auto alloca = builder.create<fir::AllocaOp>(loc, varPtr.getType());
      builder.restoreInsertionPoint(savedInsPoint);
      builder.create<fir::StoreOp>(loc, varPtr, alloca);
      varPtr = alloca;
    }
    return builder.create<omp::MapInfoOp>(
        loc, varPtr.getType(), varPtr,
        TypeAttr::get(llvm::cast<omp::PointerLikeType>(varPtr.getType())
                          .getElementType()),
        /*varPtrPtr=*/Value{},
        /*members=*/SmallVector<Value>{},
        /*member_index=*/mlir::ArrayAttr{},
        /*bounds=*/ValueRange{},
        builder.getIntegerAttr(builder.getIntegerType(64, /*isSigned=*/false),
                               mapTypeTo),
        /*mapperId*/ mlir::FlatSymbolRefAttr(),
        builder.getAttr<omp::VariableCaptureKindAttr>(
            omp::VariableCaptureKind::ByRef),
        StringAttr(), builder.getBoolAttr(false));
  }
  void addMapInfoOp(omp::TargetOp targetOp, omp::MapInfoOp mapInfoOp) {
    auto argIface = llvm::cast<omp::BlockArgOpenMPOpInterface>(*targetOp);
    unsigned insertIndex =
        argIface.getMapBlockArgsStart() + argIface.numMapBlockArgs();
    targetOp.getMapVarsMutable().append(ValueRange{mapInfoOp});
    targetOp.getRegion().insertArgument(insertIndex, mapInfoOp.getType(),
                                        mapInfoOp.getLoc());
  }
  void addMapInfoOps(omp::TargetOp targetOp,
                     llvm::SmallVectorImpl<omp::MapInfoOp> &mapInfoOps) {
    for (auto mapInfoOp : mapInfoOps)
      addMapInfoOp(targetOp, mapInfoOp);
  }
  void runOnOperation() override {
    ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
    fir::KindMapping kindMap = fir::getKindMapping(module);
    fir::FirOpBuilder builder{module, std::move(kindMap)};
    llvm::DenseMap<Operation *, llvm::SmallVector<omp::MapInfoOp, 4>>
        mapInfoOpsForTarget;

    getOperation()->walk([&](omp::TargetOp targetOp) {
      if (targetOp.getPrivateVars().empty())
        return;
      OperandRange privVars = targetOp.getPrivateVars();
      llvm::SmallVector<int64_t> privVarMapIdx;

      std::optional<ArrayAttr> privSyms = targetOp.getPrivateSyms();
      SmallVector<omp::MapInfoOp, 4> mapInfoOps;
      for (auto [privVar, privSym] : llvm::zip_equal(privVars, *privSyms)) {

        SymbolRefAttr privatizerName = llvm::cast<SymbolRefAttr>(privSym);
        omp::PrivateClauseOp privatizer =
            SymbolTable::lookupNearestSymbolFrom<omp::PrivateClauseOp>(
                targetOp, privatizerName);
        if (!privatizer.needsMap()) {
          privVarMapIdx.push_back(-1);
          continue;
        }

        privVarMapIdx.push_back(targetOp.getMapVars().size() +
                                mapInfoOps.size());

        builder.setInsertionPoint(targetOp);
        Location loc = targetOp.getLoc();
        omp::MapInfoOp mapInfoOp = createMapInfo(loc, privVar, builder);
        mapInfoOps.push_back(mapInfoOp);

        LLVM_DEBUG(llvm::dbgs() << "MapsForPrivatizedSymbolsPass created ->\n");
        LLVM_DEBUG(mapInfoOp.dump());
      }
      if (!mapInfoOps.empty()) {
        mapInfoOpsForTarget.insert({targetOp.getOperation(), mapInfoOps});
        targetOp.setPrivateMapsAttr(
            mlir::DenseI64ArrayAttr::get(targetOp.getContext(), privVarMapIdx));
      }
    });
    if (!mapInfoOpsForTarget.empty()) {
      for (auto &[targetOp, mapInfoOps] : mapInfoOpsForTarget) {
        addMapInfoOps(static_cast<omp::TargetOp>(targetOp), mapInfoOps);
      }
    }
  }
};
} // namespace
