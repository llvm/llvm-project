//===-- FIROpenACCOpsInterfaces.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of external operation interfaces for FIR.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/OpenACC/Support/FIROpenACCOpsInterfaces.h"

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"

namespace fir::acc {

template <>
mlir::Value PartialEntityAccessModel<fir::ArrayCoorOp>::getBaseEntity(
    mlir::Operation *op) const {
  return mlir::cast<fir::ArrayCoorOp>(op).getMemref();
}

template <>
mlir::Value PartialEntityAccessModel<fir::CoordinateOp>::getBaseEntity(
    mlir::Operation *op) const {
  return mlir::cast<fir::CoordinateOp>(op).getRef();
}

template <>
mlir::Value PartialEntityAccessModel<hlfir::DesignateOp>::getBaseEntity(
    mlir::Operation *op) const {
  return mlir::cast<hlfir::DesignateOp>(op).getMemref();
}

mlir::Value PartialEntityAccessModel<fir::DeclareOp>::getBaseEntity(
    mlir::Operation *op) const {
  return mlir::cast<fir::DeclareOp>(op).getStorage();
}

bool PartialEntityAccessModel<fir::DeclareOp>::isCompleteView(
    mlir::Operation *op) const {
  // Return false (partial view) only if storage is present
  // Return true (complete view) if storage is absent
  return !getBaseEntity(op);
}

mlir::Value PartialEntityAccessModel<hlfir::DeclareOp>::getBaseEntity(
    mlir::Operation *op) const {
  return mlir::cast<hlfir::DeclareOp>(op).getStorage();
}

bool PartialEntityAccessModel<hlfir::DeclareOp>::isCompleteView(
    mlir::Operation *op) const {
  // Return false (partial view) only if storage is present
  // Return true (complete view) if storage is absent
  return !getBaseEntity(op);
}

mlir::SymbolRefAttr AddressOfGlobalModel::getSymbol(mlir::Operation *op) const {
  return mlir::cast<fir::AddrOfOp>(op).getSymbolAttr();
}

bool GlobalVariableModel::isConstant(mlir::Operation *op) const {
  auto globalOp = mlir::cast<fir::GlobalOp>(op);
  return globalOp.getConstant().has_value();
}

} // namespace fir::acc
