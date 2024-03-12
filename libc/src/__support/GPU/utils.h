//===---------------- Implementation of GPU utils ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_GPU_UTILS_H
#define LLVM_LIBC_SRC___SUPPORT_GPU_UTILS_H

#include "src/__support/macros/properties/architectures.h"

#if defined(LIBC_TARGET_ARCH_IS_AMDGPU)
#include "amdgpu/utils.h"
#elif defined(LIBC_TARGET_ARCH_IS_NVPTX)
#include "nvptx/utils.h"
#else
#include "generic/utils.h"
#endif

namespace LIBC_NAMESPACE {
namespace gpu {
/// Get the first active thread inside the lane.
LIBC_INLINE uint64_t get_first_lane_id(uint64_t lane_mask) {
  return __builtin_ffsl(lane_mask) - 1;
}

/// Conditional that is only true for a single thread in a lane.
LIBC_INLINE bool is_first_lane(uint64_t lane_mask) {
  return gpu::get_lane_id() == get_first_lane_id(lane_mask);
}

/// Gets the sum of all lanes inside the warp or wavefront.
LIBC_INLINE uint32_t reduce(uint64_t lane_mask, uint32_t x) {
  for (uint32_t step = gpu::get_lane_size() / 2; step > 0; step /= 2) {
    uint32_t index = step + gpu::get_lane_id();
    x += gpu::shuffle(lane_mask, index, x);
  }
  return gpu::broadcast_value(lane_mask, x);
}

/// Gets the accumulator scan of the threads in the warp or wavefront.
LIBC_INLINE uint32_t scan(uint64_t lane_mask, uint32_t x) {
  for (uint32_t step = 1; step < gpu::get_lane_size(); step *= 2) {
    uint32_t index = gpu::get_lane_id() - step;
    uint32_t bitmask = gpu::get_lane_id() >= step;
    x += -bitmask & gpu::shuffle(lane_mask, index, x);
  }
  return x;
}

} // namespace gpu
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_GPU_UTILS_H
