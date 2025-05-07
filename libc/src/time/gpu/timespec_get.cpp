//===-- Implementation of timespec_get for gpu ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/timespec_get.h"
#include "hdr/time_macros.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/time/gpu/time_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, timespec_get, (struct timespec * ts, int base)) {
  if (base != TIME_MONOTONIC || !ts)
    return 0;

  uint64_t ns_per_tick = TICKS_PER_SEC / GPU_CLOCKS_PER_SEC;
  uint64_t ticks = gpu::fixed_frequency_clock();

  ts->tv_nsec = (ticks * ns_per_tick) % TICKS_PER_SEC;
  ts->tv_sec = (ticks * ns_per_tick) / TICKS_PER_SEC;

  return base;
}

} // namespace LIBC_NAMESPACE_DECL
