//===-- GPU implementation of the clock function --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/clock.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/time/gpu/time_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(clock_t, clock, ()) {
  if (!GPU_CLOCKS_PER_SEC)
    return clock_t(0);

  uint64_t ticks = gpu::fixed_frequency_clock();

  // We need to convert between the GPU's fixed frequency and whatever `time.h`
  // declares it to be. This is done so that dividing the result of this
  // function by 'CLOCKS_PER_SEC' yields the elapsed time.
  if (GPU_CLOCKS_PER_SEC > CLOCKS_PER_SEC)
    return clock_t(ticks / (GPU_CLOCKS_PER_SEC / CLOCKS_PER_SEC));
  return clock_t(ticks * (CLOCKS_PER_SEC / GPU_CLOCKS_PER_SEC));
}

} // namespace LIBC_NAMESPACE_DECL
