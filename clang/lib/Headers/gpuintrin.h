//===-- gpuintrin.h - Generic GPU intrinsic functions ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPUINTRIN_H
#define __GPUINTRIN_H

#if defined(__NVPTX__)
#include <nvptxintrin.h>
#elif defined(__AMDGPU__)
#include <amdgpuintrin.h>
#endif

// Returns the total number of blocks / workgroups.
_DEFAULT_ATTRS static inline uint64_t __gpu_num_blocks() {
  return __gpu_num_blocks_x() * __gpu_num_blocks_y() * __gpu_num_blocks_z();
}

// Returns the absolute id of the block / workgroup.
_DEFAULT_ATTRS static inline uint64_t __gpu_block_id() {
  return __gpu_block_id_x() +
         (uint64_t)__gpu_num_blocks_x() * __gpu_block_id_y() +
         (uint64_t)__gpu_num_blocks_x() * __gpu_num_blocks_y() *
             __gpu_block_id_z();
}

// Returns the total number of threads in the block / workgroup.
_DEFAULT_ATTRS static inline uint32_t __gpu_num_threads() {
  return __gpu_num_threads_x() * __gpu_num_threads_y() * __gpu_num_threads_z();
}

// Returns the absolute id of the thread in the current block / workgroup.
_DEFAULT_ATTRS static inline uint32_t __gpu_thread_id() {
  return __gpu_thread_id_x() + __gpu_num_threads_x() * __gpu_thread_id_y() +
         __gpu_num_threads_x() * __gpu_num_threads_y() * __gpu_thread_id_z();
}

// Get the first active thread inside the lane.
_DEFAULT_ATTRS static inline uint64_t
__gpu_first_lane_id(uint64_t __lane_mask) {
  return __builtin_ffsll(__lane_mask) - 1;
}

// Conditional that is only true for a single thread in a lane.
_DEFAULT_ATTRS static inline bool __gpu_is_first_lane(uint64_t __lane_mask) {
  return __gpu_lane_id() == __gpu_first_lane_id(__lane_mask);
}

// Gets the sum of all lanes inside the warp or wavefront.
_DEFAULT_ATTRS static inline uint32_t __gpu_lane_reduce(uint64_t __lane_mask,
                                                        uint32_t x) {
  for (uint32_t step = __gpu_num_lanes() / 2; step > 0; step /= 2) {
    uint32_t index = step + __gpu_lane_id();
    x += __gpu_shuffle_idx(__lane_mask, index, x);
  }
  return __gpu_broadcast(__lane_mask, x);
}

// Gets the accumulator scan of the threads in the warp or wavefront.
_DEFAULT_ATTRS static inline uint32_t __gpu_lane_scan(uint64_t __lane_mask,
                                                      uint32_t x) {
  for (uint32_t step = 1; step < __gpu_num_lanes(); step *= 2) {
    uint32_t index = __gpu_lane_id() - step;
    uint32_t bitmask = __gpu_lane_id() >= step;
    x += -bitmask & __gpu_shuffle_idx(__lane_mask, index, x);
  }
  return x;
}

#undef _DEFAULT_ATTRS

#endif // __GPUINTRIN_H
