//===- ehframe_registration.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains code required to load the rest of the MachO runtime.
//
//===----------------------------------------------------------------------===//

#include "adt.h"
#include "c_api.h"
#include "common.h"

using namespace __orc_rt;

// eh-frame registration functions.
// We expect these to be available for all processes.
extern "C" void __register_frame(const void *);
extern "C" void __deregister_frame(const void *);

namespace {

template <typename HandleFDEFn>
void walkEHFrameSection(span<const char> EHFrameSection,
                        HandleFDEFn HandleFDE) {
  const char *CurCFIRecord = EHFrameSection.data();
  uint64_t Size = *reinterpret_cast<const uint32_t *>(CurCFIRecord);

  while (CurCFIRecord != EHFrameSection.end() && Size != 0) {
    const char *OffsetField = CurCFIRecord + (Size == 0xffffffff ? 12 : 4);
    if (Size == 0xffffffff)
      Size = *reinterpret_cast<const uint64_t *>(CurCFIRecord + 4) + 12;
    else
      Size += 4;
    uint32_t Offset = *reinterpret_cast<const uint32_t *>(OffsetField);

    if (Offset != 0)
      HandleFDE(CurCFIRecord);

    CurCFIRecord += Size;
    Size = *reinterpret_cast<const uint32_t *>(CurCFIRecord);
  }
}

} // end anonymous namespace

ORC_RT_INTERFACE __orc_rt_CWrapperFunctionResult
__orc_rt_macho_register_ehframe_section(char *ArgData, size_t ArgSize) {
  // NOTE: Does not use SPS to deserialize arg buffer, instead the arg buffer
  // is taken to be the range of the eh-frame section.
  bool HasError = false;
  walkEHFrameSection(span<const char>(ArgData, ArgSize), __register_frame);
  return __orc_rt_CreateCWrapperFunctionResultFromRange((char*)&HasError,
                                                        sizeof(HasError));
}

ORC_RT_INTERFACE __orc_rt_CWrapperFunctionResult
__orc_rt_macho_deregister_ehframe_section(char *ArgData, size_t ArgSize) {
  // NOTE: Does not use SPS to deserialize arg buffer, instead the arg buffer
  // is taken to be the range of the eh-frame section.
  bool HasError = false;
  walkEHFrameSection(span<const char>(ArgData, ArgSize), __deregister_frame);
  return __orc_rt_CreateCWrapperFunctionResultFromRange((char*)&HasError,
                                                        sizeof(HasError));
}
