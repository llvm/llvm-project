//===- comgr-device-libs.cpp - Handle AMD Device Libraries ----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the handling of the AMD Device Libraries, which are
/// LLVM IR objects embedded into Comgr via header files.
///
/// We also handle OpenCL pre-compiled headers, which are similarly embedded in
/// Comgr.
///
//===----------------------------------------------------------------------===//

#include "comgr-device-libs.h"
#include "comgr.h"
#include "llvm/ADT/StringSwitch.h"
#include <cstdint>

using namespace llvm;

namespace COMGR {

namespace {
#include "libraries.inc"
#include "libraries_sha.inc"
#include "opencl-c-base.inc"
} // namespace

ArrayRef<unsigned char> getDeviceLibrariesIdentifier() {
  return DEVICE_LIBS_ID;
}

StringRef getOpenCLCBaseHeaderContents() {
  return StringRef(reinterpret_cast<const char *>(opencl_c_base),
                   opencl_c_base_size);
}

llvm::ArrayRef<std::tuple<llvm::StringRef, llvm::StringRef>>
getDeviceLibraries() {
  static std::tuple<llvm::StringRef, llvm::StringRef> DeviceLibs[] = {
#define AMD_DEVICE_LIBS_TARGET(target)                                         \
  {#target ".bc",                                                              \
   llvm::StringRef(reinterpret_cast<const char *>(target##_lib),               \
                   target##_lib_size)},
#include "libraries_defs.inc"
  };
  return DeviceLibs;
}

} // namespace COMGR
