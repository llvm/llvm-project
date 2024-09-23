//===- MapsForPrivatizedSymbols.cpp
//-----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "flang/Optimizer/Dialect/FIRType.h"
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

  bool privatizerNeedsMap(omp::PrivateClauseOp &privatizer) {
    Region &allocRegion = privatizer.getAllocRegion();
    Value blockArg0 = allocRegion.getArgument(0);
    if (blockArg0.use_empty())
      return false;
    return true;
  }
  omp::MapInfoOp createMapInfo(mlir::Location loc, mlir::Value var,
                               OpBuilder &builder) {
    uint64_t mapTypeTo = static_cast<
        std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
        llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO);
    Operation *definingOp = var.getDefiningOp();
    auto declOp = llvm::dyn_cast_or_null<hlfir::DeclareOp>(definingOp);
    assert(declOp &&
           "Expected defining Op of privatized var to be hlfir.declare");
    Value varPtr = declOp.getOriginalBase();
    Value varBase = declOp.getBase();
    llvm::errs() << "varPtr = ";
    varPtr.dump();
    llvm::errs() << " type -> ";
    varPtr.getType().dump();
    llvm::errs() << "\n";
    llvm::errs() << "varBase = ";
    varBase.dump();
    llvm::errs() << " type -> ";
    varBase.getType().dump();
    llvm::errs() << "\n";
    return builder.create<omp::MapInfoOp>(
        loc, varPtr.getType(), varPtr,
        mlir::TypeAttr::get(
            llvm::cast<mlir::omp::PointerLikeType>(varPtr.getType())
                .getElementType()),
        /*varPtrPtr=*/mlir::Value{},
        /*members=*/mlir::SmallVector<mlir::Value>{},
        /*member_index=*/mlir::DenseIntElementsAttr{},
        /*bounds=*/mlir::ValueRange{},
        builder.getIntegerAttr(builder.getIntegerType(64, /*isSigned=*/false),
                               mapTypeTo),
        builder.getAttr<omp::VariableCaptureKindAttr>(
            omp::VariableCaptureKind::ByRef),
        mlir::StringAttr(), builder.getBoolAttr(false));
  }
  void addMapInfoOp(omp::TargetOp targetOp, omp::MapInfoOp mapInfoOp) {
    mlir::Location loc = targetOp.getLoc();
    llvm::errs() << "Adding mapInfoOp -> ";
    mapInfoOp.dump();
    llvm::errs() << "\n";

    targetOp.getMapVarsMutable().append(mlir::ValueRange{mapInfoOp});
    size_t numMapVars = targetOp.getMapVars().size();
    targetOp.getRegion().insertArgument(numMapVars - 1, mapInfoOp.getType(),
                                        loc);
  }
  void addMapInfoOps(omp::TargetOp targetOp,
                     llvm::SmallVectorImpl<omp::MapInfoOp> &mapInfoOps) {
    for (auto mapInfoOp : mapInfoOps)
      addMapInfoOp(targetOp, mapInfoOp);
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    OpBuilder builder(context);
    llvm::DenseMap<Operation *, llvm::SmallVector<omp::MapInfoOp, 4>>
        mapInfoOpsForTarget;
    getOperation()->walk([&](omp::TargetOp targetOp) {
      if (targetOp.getPrivateVars().empty())
        return;
      llvm::errs() << "Func is \n";
      targetOp.getOperation()->getParentOp()->getParentOp()->dump();
      llvm::errs() << "\n";
      OperandRange privVars = targetOp.getPrivateVars();
      std::optional<ArrayAttr> privSyms = targetOp.getPrivateSyms();
      SmallVector<omp::MapInfoOp, 4> mapInfoOps;
      for (auto [privVar, privSym] : llvm::zip_equal(privVars, *privSyms)) {

        SymbolRefAttr privatizerName = llvm::cast<SymbolRefAttr>(privSym);
        omp::PrivateClauseOp privatizer =
            SymbolTable::lookupNearestSymbolFrom<omp::PrivateClauseOp>(
                targetOp, privatizerName);
        llvm::errs() << "privVar = ";
        privVar.dump();
        llvm::errs() << "\n";
        llvm::errs() << "privVar.getType() = ";
        privVar.getType().dump();
        llvm::errs() << "\n";
        // assert(mlir::isa<fir::ReferenceType>(privVar.getType()) &&
        //        "Privatized variable should be a reference.");
        if (!privatizerNeedsMap(privatizer)) {
          continue;
        }
        builder.setInsertionPoint(targetOp);
        mlir::Location loc = targetOp.getLoc();
        omp::MapInfoOp mapInfoOp = createMapInfo(loc, privVar, builder);
        mapInfoOps.push_back(mapInfoOp);
        LLVM_DEBUG(llvm::dbgs() << "MapsForPrivatizedSymbolsPass created ->\n");
        LLVM_DEBUG(mapInfoOp.dump());
      }
      if (!mapInfoOps.empty()) {
        mapInfoOpsForTarget.insert({targetOp.getOperation(), mapInfoOps});
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
