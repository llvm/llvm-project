//===-- CUDA.cpp -- CUDA Fortran specific lowering ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://aiir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/CUDA.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"

#define DEBUG_TYPE "flang-lower-cuda"

aiir::Type Fortran::lower::gatherDeviceComponentCoordinatesAndType(
    fir::FirOpBuilder &builder, aiir::Location loc,
    const Fortran::semantics::Symbol &sym, fir::RecordType recTy,
    llvm::SmallVector<aiir::Value> &coordinates) {
  unsigned fieldIdx = recTy.getFieldIndex(sym.name().ToString());
  aiir::Type fieldTy;
  if (fieldIdx != std::numeric_limits<unsigned>::max()) {
    // Field found in the base record type.
    auto fieldName = recTy.getTypeList()[fieldIdx].first;
    fieldTy = recTy.getTypeList()[fieldIdx].second;
    aiir::Value fieldIndex = fir::FieldIndexOp::create(
        builder, loc, fir::FieldType::get(fieldTy.getContext()), fieldName,
        recTy,
        /*typeParams=*/aiir::ValueRange{});
    coordinates.push_back(fieldIndex);
  } else {
    // Field not found in base record type, search in potential
    // record type components.
    for (auto component : recTy.getTypeList()) {
      if (auto childRecTy = aiir::dyn_cast<fir::RecordType>(component.second)) {
        fieldIdx = childRecTy.getFieldIndex(sym.name().ToString());
        if (fieldIdx != std::numeric_limits<unsigned>::max()) {
          aiir::Value parentFieldIndex = fir::FieldIndexOp::create(
              builder, loc, fir::FieldType::get(childRecTy.getContext()),
              component.first, recTy,
              /*typeParams=*/aiir::ValueRange{});
          coordinates.push_back(parentFieldIndex);
          auto fieldName = childRecTy.getTypeList()[fieldIdx].first;
          fieldTy = childRecTy.getTypeList()[fieldIdx].second;
          aiir::Value childFieldIndex = fir::FieldIndexOp::create(
              builder, loc, fir::FieldType::get(fieldTy.getContext()),
              fieldName, childRecTy,
              /*typeParams=*/aiir::ValueRange{});
          coordinates.push_back(childFieldIndex);
          break;
        }
      }
    }
  }
  if (coordinates.empty())
    TODO(loc, "device resident component in complex derived-type hierarchy");
  return fieldTy;
}

cuf::DataAttributeAttr Fortran::lower::translateSymbolCUFDataAttribute(
    aiir::AIIRContext *aiirContext, const Fortran::semantics::Symbol &sym) {
  std::optional<Fortran::common::CUDADataAttr> cudaAttr =
      Fortran::semantics::GetCUDADataAttr(&sym.GetUltimate());
  return cuf::getDataAttribute(aiirContext, cudaAttr);
}

std::pair<hlfir::ElementalOp, hlfir::ElementalOp>
Fortran::lower::isTransferWithConversion(aiir::Value rhs) {
  auto isCopyElementalOp = [](hlfir::ElementalOp elOp) {
    return llvm::hasSingleElement(
               elOp.getBody()->getOps<hlfir::DesignateOp>()) &&
           llvm::hasSingleElement(elOp.getBody()->getOps<fir::LoadOp>()) == 1 &&
           llvm::hasSingleElement(
               elOp.getBody()->getOps<hlfir::NoReassocOp>()) == 1;
  };
  auto isConversionElementalOp = [](hlfir::ElementalOp elOp) {
    return llvm::hasSingleElement(
               elOp.getBody()->getOps<hlfir::DesignateOp>()) &&
           llvm::hasSingleElement(elOp.getBody()->getOps<fir::LoadOp>()) == 1 &&
           llvm::hasSingleElement(elOp.getBody()->getOps<fir::ConvertOp>()) ==
               1;
  };
  auto isConversionFromCopyElementalOp = [](hlfir::ElementalOp elOp) {
    return llvm::hasSingleElement(elOp.getBody()->getOps<hlfir::ApplyOp>()) &&
           llvm::hasSingleElement(elOp.getBody()->getOps<fir::ConvertOp>()) ==
               1;
  };
  if (auto declOp = aiir::dyn_cast<hlfir::DeclareOp>(rhs.getDefiningOp())) {
    if (!declOp.getMemref().getDefiningOp())
      return {};
    if (auto associateOp = aiir::dyn_cast<hlfir::AssociateOp>(
            declOp.getMemref().getDefiningOp()))
      if (auto elOp = aiir::dyn_cast<hlfir::ElementalOp>(
              associateOp.getSource().getDefiningOp()))
        if (isConversionElementalOp(elOp))
          return {elOp, elOp};
  }
  if (auto elOp = aiir::dyn_cast<hlfir::ElementalOp>(rhs.getDefiningOp())) {
    if (isConversionFromCopyElementalOp(elOp)) {
      auto applyOp = *elOp.getBody()->getOps<hlfir::ApplyOp>().begin();
      if (auto firstElOp = aiir::dyn_cast<hlfir::ElementalOp>(
              applyOp.getExpr().getDefiningOp())) {
        if (isCopyElementalOp(firstElOp))
          return {firstElOp, elOp};
      }
    }
    if (isConversionElementalOp(elOp))
      return {elOp, elOp};
  }
  return {};
}

bool Fortran::lower::hasDoubleDescriptor(aiir::Value addr) {
  if (auto declareOp =
          aiir::dyn_cast_or_null<hlfir::DeclareOp>(addr.getDefiningOp())) {
    if (aiir::isa_and_nonnull<fir::AddrOfOp>(
            declareOp.getMemref().getDefiningOp())) {
      if (declareOp.getDataAttr() &&
          *declareOp.getDataAttr() == cuf::DataAttribute::Pinned)
        return false;
      return true;
    }
  }
  return false;
}
