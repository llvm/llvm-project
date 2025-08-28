//===-- lib/Utisl/OpenMP.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Utils/OpenMP.h"

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"

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
} // namespace Fortran::utils::openmp
