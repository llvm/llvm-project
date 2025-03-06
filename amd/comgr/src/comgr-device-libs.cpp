/*******************************************************************************
 *
 * University of Illinois/NCSA
 * Open Source License
 *
 * Copyright (c) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *     * Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimers.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimers in the
 *       documentation and/or other materials provided with the distribution.
 *
 *     * Neither the names of Advanced Micro Devices, Inc. nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this Software without specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 ******************************************************************************/

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

StringRef getDeviceLibrariesIdentifier() { return DEVICE_LIBS_ID; }

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
