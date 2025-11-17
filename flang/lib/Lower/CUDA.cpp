//===-- CUDA.cpp -- CUDA Fortran specific lowering ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/CUDA.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"

#define DEBUG_TYPE "flang-lower-cuda"

mlir::Type Fortran::lower::gatherDeviceComponentCoordinatesAndType(
    fir::FirOpBuilder &builder, mlir::Location loc,
    const Fortran::semantics::Symbol &sym, fir::RecordType recTy,
    llvm::SmallVector<mlir::Value> &coordinates) {
  unsigned fieldIdx = recTy.getFieldIndex(sym.name().ToString());
  mlir::Type fieldTy;
  if (fieldIdx != std::numeric_limits<unsigned>::max()) {
    // Field found in the base record type.
    auto fieldName = recTy.getTypeList()[fieldIdx].first;
    fieldTy = recTy.getTypeList()[fieldIdx].second;
    mlir::Value fieldIndex = fir::FieldIndexOp::create(
        builder, loc, fir::FieldType::get(fieldTy.getContext()), fieldName,
        recTy,
        /*typeParams=*/mlir::ValueRange{});
    coordinates.push_back(fieldIndex);
  } else {
    // Field not found in base record type, search in potential
    // record type components.
    for (auto component : recTy.getTypeList()) {
      if (auto childRecTy = mlir::dyn_cast<fir::RecordType>(component.second)) {
        fieldIdx = childRecTy.getFieldIndex(sym.name().ToString());
        if (fieldIdx != std::numeric_limits<unsigned>::max()) {
          mlir::Value parentFieldIndex = fir::FieldIndexOp::create(
              builder, loc, fir::FieldType::get(childRecTy.getContext()),
              component.first, recTy,
              /*typeParams=*/mlir::ValueRange{});
          coordinates.push_back(parentFieldIndex);
          auto fieldName = childRecTy.getTypeList()[fieldIdx].first;
          fieldTy = childRecTy.getTypeList()[fieldIdx].second;
          mlir::Value childFieldIndex = fir::FieldIndexOp::create(
              builder, loc, fir::FieldType::get(fieldTy.getContext()),
              fieldName, childRecTy,
              /*typeParams=*/mlir::ValueRange{});
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
    mlir::MLIRContext *mlirContext, const Fortran::semantics::Symbol &sym) {
  std::optional<Fortran::common::CUDADataAttr> cudaAttr =
      Fortran::semantics::GetCUDADataAttr(&sym.GetUltimate());
  return cuf::getDataAttribute(mlirContext, cudaAttr);
}

hlfir::ElementalOp Fortran::lower::isTransferWithConversion(mlir::Value rhs) {
  auto isConversionElementalOp = [](hlfir::ElementalOp elOp) {
    return llvm::hasSingleElement(
               elOp.getBody()->getOps<hlfir::DesignateOp>()) &&
           llvm::hasSingleElement(elOp.getBody()->getOps<fir::LoadOp>()) == 1 &&
           llvm::hasSingleElement(elOp.getBody()->getOps<fir::ConvertOp>()) ==
               1;
  };
  if (auto declOp = mlir::dyn_cast<hlfir::DeclareOp>(rhs.getDefiningOp())) {
    if (!declOp.getMemref().getDefiningOp())
      return {};
    if (auto associateOp = mlir::dyn_cast<hlfir::AssociateOp>(
            declOp.getMemref().getDefiningOp()))
      if (auto elOp = mlir::dyn_cast<hlfir::ElementalOp>(
              associateOp.getSource().getDefiningOp()))
        if (isConversionElementalOp(elOp))
          return elOp;
  }
  if (auto elOp = mlir::dyn_cast<hlfir::ElementalOp>(rhs.getDefiningOp()))
    if (isConversionElementalOp(elOp))
      return elOp;
  return {};
}
