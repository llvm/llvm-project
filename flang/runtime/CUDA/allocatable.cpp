//===-- runtime/CUDA/allocatable.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/CUDA/allocatable.h"
#include "../stat.h"
#include "../terminator.h"
#include "flang/Runtime/CUDA/common.h"
#include "flang/Runtime/CUDA/descriptor.h"
#include "flang/Runtime/allocatable.h"

#include "cuda_runtime.h"

namespace Fortran::runtime::cuda {

extern "C" {
RT_EXT_API_GROUP_BEGIN

int RTDEF(CUFAllocatableAllocate)(Descriptor &desc, bool hasStat,
    const Descriptor *errMsg, const char *sourceFile, int sourceLine) {
  if (desc.HasAddendum()) {
    Terminator terminator{sourceFile, sourceLine};
    // TODO: This require a bit more work to set the correct type descriptor
    // address
    terminator.Crash(
        "not yet implemented: CUDA descriptor allocation with addendum");
  }
  // Perform the standard allocation.
  int stat{RTNAME(AllocatableAllocate)(
      desc, hasStat, errMsg, sourceFile, sourceLine)};
#ifndef RT_DEVICE_COMPILATION
  // Descriptor synchronization is only done when the allocation is done
  // from the host.
  if (stat == StatOk) {
    void *deviceAddr{
        RTNAME(CUFGetDeviceAddress)((void *)&desc, sourceFile, sourceLine)};
    RTNAME(CUFDescriptorSync)
    ((Descriptor *)deviceAddr, &desc, sourceFile, sourceLine);
  }
#endif
  return stat;
}

int RTDEF(CUFAllocatableDeallocate)(Descriptor &desc, bool hasStat,
    const Descriptor *errMsg, const char *sourceFile, int sourceLine) {
  // Perform the standard allocation.
  int stat{RTNAME(AllocatableDeallocate)(
      desc, hasStat, errMsg, sourceFile, sourceLine)};
#ifndef RT_DEVICE_COMPILATION
  // Descriptor synchronization is only done when the deallocation is done
  // from the host.
  if (stat == StatOk) {
    void *deviceAddr{
        RTNAME(CUFGetDeviceAddress)((void *)&desc, sourceFile, sourceLine)};
    RTNAME(CUFDescriptorSync)
    ((Descriptor *)deviceAddr, &desc, sourceFile, sourceLine);
  }
#endif
  return stat;
}

RT_EXT_API_GROUP_END

} // extern "C"

} // namespace Fortran::runtime::cuda
