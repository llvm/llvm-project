//===-- runtime/CUDA/pointer.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/CUDA/pointer.h"
#include "../stat.h"
#include "../terminator.h"
#include "flang/Runtime/pointer.h"

#include "cuda_runtime.h"

namespace Fortran::runtime::cuda {

extern "C" {
RT_EXT_API_GROUP_BEGIN

int RTDEF(CUFPointerAllocate)(Descriptor &desc, int64_t stream, bool hasStat,
    const Descriptor *errMsg, const char *sourceFile, int sourceLine) {
  if (desc.HasAddendum()) {
    Terminator terminator{sourceFile, sourceLine};
    // TODO: This require a bit more work to set the correct type descriptor
    // address
    terminator.Crash(
        "not yet implemented: CUDA descriptor allocation with addendum");
  }
  // Perform the standard allocation.
  int stat{
      RTNAME(PointerAllocate)(desc, hasStat, errMsg, sourceFile, sourceLine)};
  return stat;
}

RT_EXT_API_GROUP_END

} // extern "C"

} // namespace Fortran::runtime::cuda
