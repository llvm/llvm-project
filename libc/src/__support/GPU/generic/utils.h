//===-------------- Generic implementation of GPU utils ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_GPU_GENERIC_IO_H
#define LLVM_LIBC_SRC_SUPPORT_GPU_GENERIC_IO_H

#include "src/__support/common.h"

#include <stdint.h>

namespace __llvm_libc {
namespace gpu {

constexpr const uint64_t LANE_SIZE = 1;

LIBC_INLINE uint32_t get_num_blocks_x() { return 1; }

LIBC_INLINE uint32_t get_num_blocks_y() { return 1; }

LIBC_INLINE uint32_t get_num_blocks_z() { return 1; }

LIBC_INLINE uint64_t get_num_blocks() { return 1; }

LIBC_INLINE uint32_t get_block_id_x() { return 0; }

LIBC_INLINE uint32_t get_block_id_y() { return 0; }

LIBC_INLINE uint32_t get_block_id_z() { return 0; }

LIBC_INLINE uint64_t get_block_id() { return 0; }

LIBC_INLINE uint32_t get_num_threads_x() { return 1; }

LIBC_INLINE uint32_t get_num_threads_y() { return 1; }

LIBC_INLINE uint32_t get_num_threads_z() { return 1; }

LIBC_INLINE uint64_t get_num_threads() { return 1; }

LIBC_INLINE uint32_t get_thread_id_x() { return 0; }

LIBC_INLINE uint32_t get_thread_id_y() { return 0; }

LIBC_INLINE uint32_t get_thread_id_z() { return 0; }

LIBC_INLINE uint64_t get_thread_id() { return 0; }

LIBC_INLINE uint32_t get_lane_size() { return LANE_SIZE; }

LIBC_INLINE uint32_t get_lane_id() { return 0; }

LIBC_INLINE uint64_t get_lane_mask() { return 1; }

LIBC_INLINE uint32_t broadcast_value(uint32_t x) { return x; }

LIBC_INLINE uint64_t ballot(uint64_t lane_mask, bool x) {
  (void)lane_mask;
  return x;
}

LIBC_INLINE void sync_threads() {}

LIBC_INLINE void sync_lane(uint64_t) {}

LIBC_INLINE uint64_t processor_clock() { return 0; }

LIBC_INLINE uint64_t fixed_frequency_clock() { return 0; }

} // namespace gpu
} // namespace __llvm_libc

#endif
