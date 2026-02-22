//===-- runtime/coarray.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/coarray.h"
#include "flang-rt/runtime/descriptor.h"
#include "flang-rt/runtime/terminator.h"
#include "flang-rt/runtime/type-info.h"

namespace Fortran::runtime {

extern "C" {
RT_EXT_API_GROUP_BEGIN

void RTDEF(ComputeLastUcobound)(
    int num_images, const Descriptor &lcobounds, const Descriptor &ucobounds) {
  int corank = ucobounds.GetDimension(0).Extent();
  if (corank > 15)
    Fortran::runtime::Terminator{}.Crash(
        "Fortran runtime error: maximum corank for a coarray is 15, current "
        "corank is %d.",
        corank);

  int64_t *lcobounds_ptr = (int64_t *)lcobounds.raw().base_addr;
  int64_t *ucobounds_ptr = (int64_t *)ucobounds.raw().base_addr;
  int64_t index = 1;
  for (int i = 0; i < corank - 1; i++) {
    index *= ucobounds_ptr[i] - lcobounds_ptr[i] + 1;
  }
  if (corank == 1)
    ucobounds_ptr[0] = lcobounds_ptr[0] + num_images - 1;
  else if (index < num_images)
    ucobounds_ptr[corank - 1] = lcobounds_ptr[corank - 1] + 
        (num_images / index) + (num_images % index != 0) - 1;
  else
    ucobounds_ptr[corank - 1] = lcobounds_ptr[corank - 1];
}

RT_EXT_API_GROUP_END
}
} // namespace Fortran::runtime
