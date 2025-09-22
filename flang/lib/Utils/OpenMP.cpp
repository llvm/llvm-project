//===-- lib/Utisl/OpenMP.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Utils/OpenMP.h"

#include "flang/Lower/ConvertExprToHLFIR.h"
#include "flang/Optimizer/Builder/DirectivesCommon.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Transforms/RegionUtils.h"

namespace Fortran::utils::openmp {
mlir::omp::MapInfoOp createMapInfoOp(mlir::OpBuilder &builder,
    mlir::Location loc, mlir::Value baseAddr, mlir::Value varPtrPtr,
    llvm::StringRef name, llvm::ArrayRef<mlir::Value> bounds,
    llvm::ArrayRef<mlir::Value> members, mlir::ArrayAttr membersIndex,
    uint64_t mapType, mlir::omp::VariableCaptureKind mapCaptureType,
    mlir::Type retTy, bool partialMap, mlir::FlatSymbolRefAttr mapperId) {

  if (auto boxTy = llvm::dyn_cast<fir::BaseBoxType>(baseAddr.getType())) {
    baseAddr = fir::BoxAddrOp::create(builder, loc, baseAddr);
    retTy = baseAddr.getType();
  }

  mlir::TypeAttr varType = mlir::TypeAttr::get(
      llvm::cast<mlir::omp::PointerLikeType>(retTy).getElementType());

  // For types with unknown extents such as <2x?xi32> we discard the incomplete
  // type info and only retain the base type. The correct dimensions are later
  // recovered through the bounds info.
  if (auto seqType = llvm::dyn_cast<fir::SequenceType>(varType.getValue()))
    if (seqType.hasDynamicExtents())
      varType = mlir::TypeAttr::get(seqType.getEleTy());

  mlir::omp::MapInfoOp op =
      mlir::omp::MapInfoOp::create(builder, loc, retTy, baseAddr, varType,
          builder.getIntegerAttr(builder.getIntegerType(64, false), mapType),
          builder.getAttr<mlir::omp::VariableCaptureKindAttr>(mapCaptureType),
          varPtrPtr, members, membersIndex, bounds, mapperId,
          builder.getStringAttr(name), builder.getBoolAttr(partialMap));
  return op;
}

mlir::Value mapTemporaryValue(fir::FirOpBuilder &firOpBuilder,
    mlir::omp::TargetOp targetOp, mlir::Value val, llvm::StringRef name) {
  mlir::OpBuilder::InsertionGuard guard(firOpBuilder);
  mlir::Operation *valOp = val.getDefiningOp();

  if (valOp)
    firOpBuilder.setInsertionPointAfter(valOp);
  else
    // This means val is a block argument
    firOpBuilder.setInsertionPoint(targetOp);

  auto copyVal = firOpBuilder.createTemporary(val.getLoc(), val.getType());
  firOpBuilder.createStoreWithConvert(copyVal.getLoc(), val, copyVal);

  fir::factory::AddrAndBoundsInfo info = fir::factory::getDataOperandBaseAddr(
      firOpBuilder, val, /*isOptional=*/false, val.getLoc());
  llvm::SmallVector<mlir::Value> bounds =
      fir::factory::genImplicitBoundsOps<mlir::omp::MapBoundsOp,
          mlir::omp::MapBoundsType>(firOpBuilder, info,
          hlfir::translateToExtendedValue(
              val.getLoc(), firOpBuilder, hlfir::Entity{val})
              .first,
          /*dataExvIsAssumedSize=*/false, val.getLoc());

  firOpBuilder.setInsertionPoint(targetOp);

  llvm::omp::OpenMPOffloadMappingFlags mapFlag =
      llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_IMPLICIT;
  mlir::omp::VariableCaptureKind captureKind =
      mlir::omp::VariableCaptureKind::ByRef;

  mlir::Type eleType = copyVal.getType();
  if (auto refType = mlir::dyn_cast<fir::ReferenceType>(copyVal.getType())) {
    eleType = refType.getElementType();
  }

  if (fir::isa_trivial(eleType) || fir::isa_char(eleType)) {
    captureKind = mlir::omp::VariableCaptureKind::ByCopy;
  } else if (!fir::isa_builtin_cptr_type(eleType)) {
    mapFlag |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO;
  }

  mlir::Value mapOp = createMapInfoOp(firOpBuilder, copyVal.getLoc(), copyVal,
      /*varPtrPtr=*/mlir::Value{}, name.str(), bounds,
      /*members=*/llvm::SmallVector<mlir::Value>{},
      /*membersIndex=*/mlir::ArrayAttr{},
      static_cast<std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
          mapFlag),
      captureKind, copyVal.getType());

  auto argIface = llvm::cast<mlir::omp::BlockArgOpenMPOpInterface>(*targetOp);
  mlir::Region &region = targetOp.getRegion();

  // Get the index of the first non-map argument before modifying mapVars,
  // then append an element to mapVars and an associated entry block
  // argument at that index.
  unsigned insertIndex =
      argIface.getMapBlockArgsStart() + argIface.numMapBlockArgs();
  targetOp.getMapVarsMutable().append(mapOp);
  mlir::Value clonedValArg =
      region.insertArgument(insertIndex, copyVal.getType(), copyVal.getLoc());

  mlir::Block *entryBlock = &region.getBlocks().front();
  firOpBuilder.setInsertionPointToStart(entryBlock);
  auto loadOp =
      firOpBuilder.create<fir::LoadOp>(clonedValArg.getLoc(), clonedValArg);
  return loadOp.getResult();
}

void cloneOrMapRegionOutsiders(
    fir::FirOpBuilder &firOpBuilder, mlir::omp::TargetOp targetOp) {
  mlir::Region &region = targetOp.getRegion();
  mlir::Block *entryBlock = &region.getBlocks().front();

  llvm::SetVector<mlir::Value> valuesDefinedAbove;
  mlir::getUsedValuesDefinedAbove(region, valuesDefinedAbove);
  while (!valuesDefinedAbove.empty()) {
    for (mlir::Value val : valuesDefinedAbove) {
      mlir::Operation *valOp = val.getDefiningOp();

      // NOTE: We skip BoxDimsOp's as the lesser of two evils is to map the
      // indices separately, as the alternative is to eventually map the Box,
      // which comes with a fairly large overhead comparatively. We could be
      // more robust about this and check using a BackwardsSlice to see if we
      // run the risk of mapping a box.
      if (valOp && mlir::isMemoryEffectFree(valOp) &&
          !mlir::isa<fir::BoxDimsOp>(valOp)) {
        mlir::Operation *clonedOp = valOp->clone();
        entryBlock->push_front(clonedOp);

        auto replace = [entryBlock](mlir::OpOperand &use) {
          return use.getOwner()->getBlock() == entryBlock;
        };

        valOp->getResults().replaceUsesWithIf(clonedOp->getResults(), replace);
        valOp->replaceUsesWithIf(clonedOp, replace);
      } else {
        mlir::Value mappedTemp = mapTemporaryValue(firOpBuilder, targetOp, val,
            /*name=*/{});
        val.replaceUsesWithIf(mappedTemp, [entryBlock](mlir::OpOperand &use) {
          return use.getOwner()->getBlock() == entryBlock;
        });
      }
    }
    valuesDefinedAbove.clear();
    mlir::getUsedValuesDefinedAbove(region, valuesDefinedAbove);
  }
}
} // namespace Fortran::utils::openmp
