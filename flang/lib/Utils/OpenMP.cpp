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

#include "aiir/Dialect/OpenMP/OpenMPDialect.h"
#include "aiir/Transforms/RegionUtils.h"

namespace Fortran::utils::openmp {
aiir::omp::MapInfoOp createMapInfoOp(aiir::OpBuilder &builder,
    aiir::Location loc, aiir::Value baseAddr, aiir::Value varPtrPtr,
    llvm::StringRef name, llvm::ArrayRef<aiir::Value> bounds,
    llvm::ArrayRef<aiir::Value> members, aiir::ArrayAttr membersIndex,
    aiir::omp::ClauseMapFlags mapType,
    aiir::omp::VariableCaptureKind mapCaptureType, aiir::Type retTy,
    bool partialMap, aiir::FlatSymbolRefAttr mapperId) {

  if (auto boxTy = llvm::dyn_cast<fir::BaseBoxType>(baseAddr.getType())) {
    baseAddr = fir::BoxAddrOp::create(builder, loc, baseAddr);
    retTy = baseAddr.getType();
  }

  aiir::TypeAttr varType = aiir::TypeAttr::get(
      llvm::cast<aiir::omp::PointerLikeType>(retTy).getElementType());

  // For types with unknown extents such as <2x?xi32> we discard the incomplete
  // type info and only retain the base type. The correct dimensions are later
  // recovered through the bounds info.
  if (auto seqType = llvm::dyn_cast<fir::SequenceType>(varType.getValue()))
    if (seqType.hasDynamicExtents())
      varType = aiir::TypeAttr::get(seqType.getEleTy());

  aiir::omp::MapInfoOp op =
      aiir::omp::MapInfoOp::create(builder, loc, retTy, baseAddr, varType,
          builder.getAttr<aiir::omp::ClauseMapFlagsAttr>(mapType),
          builder.getAttr<aiir::omp::VariableCaptureKindAttr>(mapCaptureType),
          varPtrPtr, members, membersIndex, bounds, mapperId,
          builder.getStringAttr(name), builder.getBoolAttr(partialMap));
  return op;
}

aiir::Value mapTemporaryValue(fir::FirOpBuilder &firOpBuilder,
    aiir::omp::TargetOp targetOp, aiir::Value val, llvm::StringRef name) {
  aiir::OpBuilder::InsertionGuard guard(firOpBuilder);
  aiir::Operation *valOp = val.getDefiningOp();

  if (valOp)
    firOpBuilder.setInsertionPointAfter(valOp);
  else
    // This means val is a block argument
    firOpBuilder.setInsertionPoint(targetOp);

  auto copyVal = firOpBuilder.createTemporary(val.getLoc(), val.getType());
  firOpBuilder.createStoreWithConvert(copyVal.getLoc(), val, copyVal);

  fir::factory::AddrAndBoundsInfo info = fir::factory::getDataOperandBaseAddr(
      firOpBuilder, val, /*isOptional=*/false, val.getLoc());
  llvm::SmallVector<aiir::Value> bounds =
      fir::factory::genImplicitBoundsOps<aiir::omp::MapBoundsOp,
          aiir::omp::MapBoundsType>(firOpBuilder, info,
          hlfir::translateToExtendedValue(
              val.getLoc(), firOpBuilder, hlfir::Entity{val})
              .first,
          /*dataExvIsAssumedSize=*/false, val.getLoc());

  firOpBuilder.setInsertionPoint(targetOp);

  aiir::omp::ClauseMapFlags mapFlag = aiir::omp::ClauseMapFlags::implicit;
  aiir::omp::VariableCaptureKind captureKind =
      aiir::omp::VariableCaptureKind::ByRef;

  aiir::Type eleType = copyVal.getType();
  if (auto refType = aiir::dyn_cast<fir::ReferenceType>(copyVal.getType())) {
    eleType = refType.getElementType();
  }

  if (fir::isa_trivial(eleType) || fir::isa_char(eleType)) {
    captureKind = aiir::omp::VariableCaptureKind::ByCopy;
  } else if (!fir::isa_builtin_cptr_type(eleType)) {
    mapFlag |= aiir::omp::ClauseMapFlags::to;
  }

  aiir::Value mapOp = createMapInfoOp(firOpBuilder, copyVal.getLoc(), copyVal,
      /*varPtrPtr=*/aiir::Value{}, name.str(), bounds,
      /*members=*/llvm::SmallVector<aiir::Value>{},
      /*membersIndex=*/aiir::ArrayAttr{}, mapFlag, captureKind,
      copyVal.getType());

  auto argIface = llvm::cast<aiir::omp::BlockArgOpenMPOpInterface>(*targetOp);
  aiir::Region &region = targetOp.getRegion();

  // Get the index of the first non-map argument before modifying mapVars,
  // then append an element to mapVars and an associated entry block
  // argument at that index.
  unsigned insertIndex =
      argIface.getMapBlockArgsStart() + argIface.numMapBlockArgs();
  targetOp.getMapVarsMutable().append(mapOp);
  aiir::Value clonedValArg =
      region.insertArgument(insertIndex, copyVal.getType(), copyVal.getLoc());

  aiir::Block *entryBlock = &region.getBlocks().front();
  firOpBuilder.setInsertionPointToStart(entryBlock);
  auto loadOp =
      fir::LoadOp::create(firOpBuilder, clonedValArg.getLoc(), clonedValArg);
  return loadOp.getResult();
}

void cloneOrMapRegionOutsiders(
    fir::FirOpBuilder &firOpBuilder, aiir::omp::TargetOp targetOp) {
  aiir::Region &region = targetOp.getRegion();
  aiir::Block *entryBlock = &region.getBlocks().front();

  llvm::SetVector<aiir::Value> valuesDefinedAbove;
  aiir::getUsedValuesDefinedAbove(region, valuesDefinedAbove);
  while (!valuesDefinedAbove.empty()) {
    for (aiir::Value val : valuesDefinedAbove) {
      aiir::Operation *valOp = val.getDefiningOp();

      // NOTE: We skip BoxDimsOp's as the lesser of two evils is to map the
      // indices separately, as the alternative is to eventually map the Box,
      // which comes with a fairly large overhead comparatively. We could be
      // more robust about this and check using a BackwardsSlice to see if we
      // run the risk of mapping a box.
      if (valOp && aiir::isMemoryEffectFree(valOp) &&
          !aiir::isa<fir::BoxDimsOp>(valOp)) {
        aiir::Operation *clonedOp = valOp->clone();
        entryBlock->push_front(clonedOp);

        auto replace = [entryBlock](aiir::OpOperand &use) {
          return use.getOwner()->getBlock() == entryBlock;
        };

        valOp->getResults().replaceUsesWithIf(clonedOp->getResults(), replace);
        valOp->replaceUsesWithIf(clonedOp, replace);
      } else {
        aiir::Value mappedTemp = mapTemporaryValue(firOpBuilder, targetOp, val,
            /*name=*/{});
        val.replaceUsesWithIf(mappedTemp, [entryBlock](aiir::OpOperand &use) {
          return use.getOwner()->getBlock() == entryBlock;
        });
      }
    }
    valuesDefinedAbove.clear();
    aiir::getUsedValuesDefinedAbove(region, valuesDefinedAbove);
  }
}

/// Gets or generates a default declare mapper for a given record type.
///
/// \param firOpBuilder The builder to use for generating the mapper.
/// \param loc The location to use for the generated operations.
/// \param recordType The record type to generate the mapper for.
/// \param mapperNameStr The name of the mapper to generate.
/// \param mangler A function to mangle the mapper name for nested types.
aiir::FlatSymbolRefAttr getOrGenImplicitDefaultDeclareMapper(
    fir::FirOpBuilder &firOpBuilder, aiir::Location loc,
    fir::RecordType recordType, llvm::StringRef mapperNameStr,
    RecordMemberMapperMangler mangler) {
  if (mapperNameStr.empty())
    return {};

  aiir::ModuleOp moduleOp = firOpBuilder.getModule();
  if (moduleOp.lookupSymbol(mapperNameStr))
    return aiir::FlatSymbolRefAttr::get(
        firOpBuilder.getContext(), mapperNameStr);

  aiir::OpBuilder::InsertionGuard guard(firOpBuilder);

  firOpBuilder.setInsertionPointToStart(moduleOp.getBody());
  auto declMapperOp = aiir::omp::DeclareMapperOp::create(
      firOpBuilder, loc, mapperNameStr, recordType);
  auto &region = declMapperOp.getRegion();
  firOpBuilder.createBlock(&region);
  auto mapperArg = region.addArgument(firOpBuilder.getRefType(recordType), loc);

  auto declareOp = hlfir::DeclareOp::create(firOpBuilder, loc, mapperArg,
      /*uniq_name=*/"");

  const auto genBoundsOps = [&](aiir::Value mapVal,
                                llvm::SmallVectorImpl<aiir::Value> &bounds) {
    fir::ExtendedValue extVal = hlfir::translateToExtendedValue(mapVal.getLoc(),
        firOpBuilder, hlfir::Entity{mapVal},
        /*contiguousHint=*/true)
                                    .first;
    fir::factory::AddrAndBoundsInfo info = fir::factory::getDataOperandBaseAddr(
        firOpBuilder, mapVal, /*isOptional=*/false, mapVal.getLoc());
    bounds = fir::factory::genImplicitBoundsOps<aiir::omp::MapBoundsOp,
        aiir::omp::MapBoundsType>(firOpBuilder, info, extVal,
        /*dataExvIsAssumedSize=*/false, mapVal.getLoc());
  };

  const auto getFieldRef = [&](aiir::Value rec, llvm::StringRef fieldName,
                               aiir::Type fieldTy, aiir::Type recType) {
    aiir::Value field = fir::FieldIndexOp::create(firOpBuilder, loc,
        fir::FieldType::get(recType.getContext()), fieldName, recType,
        fir::getTypeParams(rec));
    return fir::CoordinateOp::create(
        firOpBuilder, loc, firOpBuilder.getRefType(fieldTy), rec, field);
  };

  llvm::SmallVector<aiir::Value> clauseMapVars;
  llvm::SmallVector<llvm::SmallVector<int64_t>> memberPlacementIndices;
  llvm::SmallVector<aiir::Value> memberMapOps;

  aiir::omp::ClauseMapFlags mapFlag = aiir::omp::ClauseMapFlags::to |
      aiir::omp::ClauseMapFlags::from | aiir::omp::ClauseMapFlags::implicit;
  aiir::omp::VariableCaptureKind captureKind =
      aiir::omp::VariableCaptureKind::ByRef;

  for (const auto &entry : llvm::enumerate(recordType.getTypeList())) {
    const auto &memberName = entry.value().first;
    const auto &memberType = entry.value().second;
    aiir::FlatSymbolRefAttr mapperId;
    if (auto recType = aiir::dyn_cast<fir::RecordType>(
            fir::getFortranElementType(memberType))) {
      std::string mapperIdName =
          recType.getName().str() + llvm::omp::OmpDefaultMapperName;
      mangler(mapperIdName, memberName);
      mapperId = getOrGenImplicitDefaultDeclareMapper(
          firOpBuilder, loc, recType, mapperIdName, mangler);
    }

    auto ref =
        getFieldRef(declareOp.getBase(), memberName, memberType, recordType);
    llvm::SmallVector<aiir::Value> bounds;
    genBoundsOps(ref, bounds);
    aiir::Value mapOp = Fortran::utils::openmp::createMapInfoOp(firOpBuilder,
        loc, ref, /*varPtrPtr=*/aiir::Value{}, /*name=*/"", bounds,
        /*members=*/{},
        /*membersIndex=*/aiir::ArrayAttr{}, mapFlag, captureKind, ref.getType(),
        /*partialMap=*/false, mapperId);
    memberMapOps.emplace_back(mapOp);
    memberPlacementIndices.emplace_back(
        llvm::SmallVector<int64_t>{(int64_t)entry.index()});
  }

  llvm::SmallVector<aiir::Value> bounds;
  genBoundsOps(declareOp.getOriginalBase(), bounds);
  aiir::omp::ClauseMapFlags parentMapFlag = aiir::omp::ClauseMapFlags::implicit;
  aiir::omp::MapInfoOp mapOp = Fortran::utils::openmp::createMapInfoOp(
      firOpBuilder, loc, declareOp.getOriginalBase(),
      /*varPtrPtr=*/aiir::Value(), /*name=*/"", bounds, memberMapOps,
      firOpBuilder.create2DI64ArrayAttr(memberPlacementIndices), parentMapFlag,
      captureKind, declareOp.getType(0),
      /*partialMap=*/true);

  clauseMapVars.emplace_back(mapOp);
  aiir::omp::DeclareMapperInfoOp::create(firOpBuilder, loc, clauseMapVars);
  return aiir::FlatSymbolRefAttr::get(firOpBuilder.getContext(), mapperNameStr);
}
} // namespace Fortran::utils::openmp
