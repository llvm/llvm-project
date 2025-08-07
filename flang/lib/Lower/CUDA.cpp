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

#define DEBUG_TYPE "flang-lower-cuda"

void Fortran::lower::initializeDeviceComponentAllocator(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::semantics::Symbol &sym, const fir::MutableBoxValue &box) {
  if (const auto *details{
          sym.GetUltimate()
              .detailsIf<Fortran::semantics::ObjectEntityDetails>()}) {
    const Fortran::semantics::DeclTypeSpec *type{details->type()};
    const Fortran::semantics::DerivedTypeSpec *derived{type ? type->AsDerived()
                                                            : nullptr};
    if (derived) {
      if (!FindCUDADeviceAllocatableUltimateComponent(*derived))
        return; // No device components.

      fir::FirOpBuilder &builder = converter.getFirOpBuilder();
      mlir::Location loc = converter.getCurrentLocation();

      mlir::Type baseTy = fir::unwrapRefType(box.getAddr().getType());

      // Only pointer and allocatable needs post allocation initialization
      // of components descriptors.
      if (!fir::isAllocatableType(baseTy) && !fir::isPointerType(baseTy))
        return;

      // Extract the derived type.
      mlir::Type ty = fir::getDerivedType(baseTy);
      auto recTy = mlir::dyn_cast<fir::RecordType>(ty);
      assert(recTy && "expected fir::RecordType");

      if (auto boxTy = mlir::dyn_cast<fir::BaseBoxType>(baseTy))
        baseTy = boxTy.getEleTy();
      baseTy = fir::unwrapRefType(baseTy);

      Fortran::semantics::UltimateComponentIterator components{*derived};
      mlir::Value loadedBox = fir::LoadOp::create(builder, loc, box.getAddr());
      mlir::Value addr;
      if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(baseTy)) {
        mlir::Type idxTy = builder.getIndexType();
        mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
        mlir::Value zero = builder.createIntegerConstant(loc, idxTy, 0);
        llvm::SmallVector<fir::DoLoopOp> loops;
        llvm::SmallVector<mlir::Value> indices;
        llvm::SmallVector<mlir::Value> extents;
        for (unsigned i = 0; i < seqTy.getDimension(); ++i) {
          mlir::Value dim = builder.createIntegerConstant(loc, idxTy, i);
          auto dimInfo = fir::BoxDimsOp::create(builder, loc, idxTy, idxTy,
                                                idxTy, loadedBox, dim);
          mlir::Value lbub = mlir::arith::AddIOp::create(
              builder, loc, dimInfo.getResult(0), dimInfo.getResult(1));
          mlir::Value ext =
              mlir::arith::SubIOp::create(builder, loc, lbub, one);
          mlir::Value cmp = mlir::arith::CmpIOp::create(
              builder, loc, mlir::arith::CmpIPredicate::sgt, ext, zero);
          ext = mlir::arith::SelectOp::create(builder, loc, cmp, ext, zero);
          extents.push_back(ext);

          auto loop = fir::DoLoopOp::create(
              builder, loc, dimInfo.getResult(0), dimInfo.getResult(1),
              dimInfo.getResult(2), /*isUnordered=*/true,
              /*finalCount=*/false, mlir::ValueRange{});
          loops.push_back(loop);
          indices.push_back(loop.getInductionVar());
          builder.setInsertionPointToStart(loop.getBody());
        }
        mlir::Value boxAddr = fir::BoxAddrOp::create(builder, loc, loadedBox);
        auto shape = fir::ShapeOp::create(builder, loc, extents);
        addr = fir::ArrayCoorOp::create(
            builder, loc, fir::ReferenceType::get(recTy), boxAddr, shape,
            /*slice=*/mlir::Value{}, indices, /*typeparms=*/mlir::ValueRange{});
      } else {
        addr = fir::BoxAddrOp::create(builder, loc, loadedBox);
      }
      for (const auto &compSym : components) {
        if (Fortran::semantics::IsDeviceAllocatable(compSym)) {
          llvm::SmallVector<mlir::Value> coord;
          mlir::Type fieldTy = gatherDeviceComponentCoordinatesAndType(
              builder, loc, compSym, recTy, coord);
          assert(coord.size() == 1 && "expect one coordinate");
          mlir::Value comp = fir::CoordinateOp::create(
              builder, loc, builder.getRefType(fieldTy), addr, coord[0]);
          cuf::DataAttributeAttr dataAttr =
              Fortran::lower::translateSymbolCUFDataAttribute(
                  builder.getContext(), compSym);
          cuf::SetAllocatorIndexOp::create(builder, loc, comp, dataAttr);
        }
      }
    }
  }
}

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
