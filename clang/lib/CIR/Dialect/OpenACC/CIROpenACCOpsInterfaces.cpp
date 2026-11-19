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
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace cir;

static std::optional<LangAddressSpace>
getGlobalLangAddressSpace(Operation *op) {
  auto langAS = dyn_cast_if_present<LangAddressSpaceAttr>(
      cast<GlobalOp>(op).getAddrSpaceAttr());
  if (!langAS)
    return std::nullopt;
  return langAS.getValue();
}

// Address spaces OpenACC treats as reachable from the device. This includes
// memory that resides on the device and host memory the device can access
// directly (`offload_global_host`).
static bool isDeviceAccessibleAddressSpace(LangAddressSpace as) {
  switch (as) {
  case LangAddressSpace::OffloadPrivate:
  case LangAddressSpace::OffloadLocal:
  case LangAddressSpace::OffloadGlobal:
  case LangAddressSpace::OffloadConstant:
  case LangAddressSpace::OffloadGlobalDevice:
  case LangAddressSpace::OffloadGlobalHost:
    return true;
  case LangAddressSpace::Default:
  case LangAddressSpace::OffloadGeneric:
    return false;
  }
  llvm_unreachable("unhandled CIR language address space");
}

// Address spaces whose storage OpenACC treats as residing on the device.
static bool isDeviceResidentAddressSpace(LangAddressSpace as) {
  if (as == LangAddressSpace::OffloadGlobalHost)
    return false;
  return isDeviceAccessibleAddressSpace(as);
}

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
  std::optional<LangAddressSpace> as = getGlobalLangAddressSpace(op);
  return as && isDeviceAccessibleAddressSpace(*as);
}

bool cir::acc::GlobalVariableModel::isInDeviceMemory(Operation *op) const {
  std::optional<LangAddressSpace> as = getGlobalLangAddressSpace(op);
  return as && isDeviceResidentAddressSpace(*as);
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