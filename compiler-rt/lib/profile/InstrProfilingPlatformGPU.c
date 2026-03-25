/*===- InstrProfilingPlatformGPU.c - GPU profiling support ----------------===*\
|*
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
|* See https://llvm.org/LICENSE.txt for license information.
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
|*
\*===----------------------------------------------------------------------===*/

// GPU-specific profiling functions for AMDGPU and NVPTX targets. This file
// provides:
//
// Platform plumbing (section boundaries, binary IDs, VNodes) are handled by
// InstrProfilingPlatformLinux.c via the COMPILER_RT_PROFILE_BAREMETAL path.

#if defined(__NVPTX__) || defined(__AMDGPU__)

#include "InstrProfiling.h"
#include <gpuintrin.h>

// Indicates that the current wave is fully occupied.
static int is_uniform(uint64_t mask) {
  const uint64_t uniform_mask = ~0ull >> (64 - __gpu_num_lanes());
  return mask == uniform_mask;
}

// Wave-cooperative counter increment. The instrumentation pass emits calls to
// this in place of the default non-atomic load/add/store or atomicrmw sequence.
// The optional uniform counter allows calculating wave uniformity if present.
COMPILER_RT_VISIBILITY void __llvm_profile_instrument_gpu(uint64_t *counter,
                                                          uint64_t *uniform,
                                                          uint64_t step) {
  uint64_t mask = __gpu_lane_mask();
  if (__gpu_is_first_in_lane(mask)) {
    __scoped_atomic_fetch_add(counter, step * __builtin_popcountg(mask),
                              __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    if (uniform && is_uniform(mask))
      __scoped_atomic_fetch_add(uniform, step * __builtin_popcountg(mask),
                                __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  }
}

#endif
