//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of external OpenACC operation interfaces for CIR.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/OpenACC/CIROpenACCOpsInterfaces.h"

#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Dialect/OpenACC/CIROpenACCUtils.h"

using namespace mlir;
using namespace cir;

SymbolRefAttr cir::acc::AddressOfGlobalModel::getSymbol(Operation *op) const {
  return cast<GetGlobalOp>(op).getNameAttr();
}

bool cir::acc::GlobalVariableModel::isConstant(Operation *op) const {
  return cast<GlobalOp>(op).getConstant();
}

bool cir::acc::GlobalVariableModel::hasInitializer(Operation *op) const {
  return cast<GlobalOp>(op).hasInitializer();
}

Region *cir::acc::GlobalVariableModel::getInitRegion(Operation *op) const {
  Region &ctorRegion = cast<GlobalOp>(op).getCtorRegion();
  return ctorRegion.empty() ? nullptr : &ctorRegion;
}

bool cir::acc::GlobalVariableModel::isDeviceAccessible(Operation *op) const {
  auto lang = dyn_cast_if_present<LangAddressSpaceAttr>(
      cast<GlobalOp>(op).getAddrSpaceAttr());
  return lang && isDeviceAccessibleAddressSpace(lang.getValue());
}

bool cir::acc::GlobalVariableModel::isInDeviceMemory(Operation *op) const {
  auto lang = dyn_cast_if_present<LangAddressSpaceAttr>(
      cast<GlobalOp>(op).getAddrSpaceAttr());
  return lang && isDeviceResidentAddressSpace(lang.getValue());
}

// A leading '.' marks a symbol that is not a source identifier. The
// `__const.` prefix marks a materialized constant aggregate. Ideally this
// should be a property of the global itself, but having a property puts
// burden on every creation site to decide it and this is a reasonable
// approximation for now.
bool cir::acc::GlobalVariableModel::isCompilerGenerated(Operation *op) const {
  StringRef name = cast<GlobalOp>(op).getSymName();
  return name.starts_with(".") || name.starts_with("__const.");
}
