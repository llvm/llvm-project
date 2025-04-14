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
amd_comgr_status_t addObject(DataSet *DataSet, amd_comgr_data_kind_t Kind,
                             const char *Name, const void *Data, size_t Size) {
  DataObject *Obj = DataObject::allocate(Kind);
  if (!Obj) {
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  if (auto Status = Obj->setName(Name)) {
    return Status;
  }
  if (auto Status =
          Obj->setData(StringRef(reinterpret_cast<const char *>(Data), Size))) {
    return Status;
  }
  DataSet->DataObjects.insert(Obj);
  return AMD_COMGR_STATUS_SUCCESS;
}

#include "libraries.inc"
#include "libraries_sha.inc"
#include "opencl1.2-c.inc"
#include "opencl2.0-c.inc"
} // namespace

ArrayRef<unsigned char> getDeviceLibrariesIdentifier() {
  return DEVICE_LIBS_ID;
}

amd_comgr_status_t addPrecompiledHeaders(DataAction *ActionInfo,
                                         DataSet *ResultSet) {
  switch (ActionInfo->Language) {
  case AMD_COMGR_LANGUAGE_OPENCL_1_2:
    return addObject(ResultSet, AMD_COMGR_DATA_KIND_PRECOMPILED_HEADER,
                     "opencl1.2-c.pch", opencl1_2_c, opencl1_2_c_size);
  case AMD_COMGR_LANGUAGE_OPENCL_2_0:
    return addObject(ResultSet, AMD_COMGR_DATA_KIND_PRECOMPILED_HEADER,
                     "opencl2.0-c.pch", opencl2_0_c, opencl2_0_c_size);
  default:
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }
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
