//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of CIR OpenACC utilities.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/OpenACC/CIROpenACCUtils.h"

#include "llvm/Support/ErrorHandling.h"

namespace cir::acc {

bool isDeviceAccessibleAddressSpace(cir::LangAddressSpace space) {
  switch (space) {
  case cir::LangAddressSpace::OffloadPrivate:
  case cir::LangAddressSpace::OffloadLocal:
  case cir::LangAddressSpace::OffloadGlobal:
  case cir::LangAddressSpace::OffloadConstant:
  case cir::LangAddressSpace::OffloadGlobalDevice:
  case cir::LangAddressSpace::OffloadGlobalHost:
  case cir::LangAddressSpace::OffloadGeneric:
    return true;
  case cir::LangAddressSpace::Default:
    return false;
  }
  llvm_unreachable("unhandled CIR language address space");
}

bool isDeviceResidentAddressSpace(cir::LangAddressSpace space) {
  switch (space) {
  case cir::LangAddressSpace::OffloadPrivate:
  case cir::LangAddressSpace::OffloadLocal:
  case cir::LangAddressSpace::OffloadGlobal:
  case cir::LangAddressSpace::OffloadConstant:
  case cir::LangAddressSpace::OffloadGlobalDevice:
    return true;
  case cir::LangAddressSpace::Default:
  case cir::LangAddressSpace::OffloadGlobalHost:
  case cir::LangAddressSpace::OffloadGeneric:
    return false;
  }
  llvm_unreachable("unhandled CIR language address space");
}

} // namespace cir::acc
