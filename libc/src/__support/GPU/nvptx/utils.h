//===-------------- NVPTX implementation of GPU utils -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-id: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_GPU_NVPTX_IO_H
#define LLVM_LIBC_SRC___SUPPORT_GPU_NVPTX_IO_H

#include "src/__support/common.h"

#include <stdint.h>

namespace LIBC_NAMESPACE {
namespace gpu {

/// The number of threads that execute in lock-step in a warp.
constexpr const uint64_t LANE_SIZE = 32;

/// Type aliases to the address spaces used by the NVPTX backend.
template <typename T> using Private = [[clang::opencl_private]] T;
template <typename T> using Constant = [[clang::opencl_constant]] T;
template <typename T> using Local = [[clang::opencl_local]] T;
template <typename T> using Global = [[clang::opencl_global]] T;

/// Returns the number of CUDA blocks in the 'x' dimension.
LIBC_INLINE uint32_t get_num_blocks_x() {
  return __nvvm_read_ptx_sreg_nctaid_x();
}

/// Returns the number of CUDA blocks in the 'y' dimension.
LIBC_INLINE uint32_t get_num_blocks_y() {
  return __nvvm_read_ptx_sreg_nctaid_y();
}

/// Returns the number of CUDA blocks in the 'z' dimension.
LIBC_INLINE uint32_t get_num_blocks_z() {
  return __nvvm_read_ptx_sreg_nctaid_z();
}

/// Returns the total number of CUDA blocks.
LIBC_INLINE uint64_t get_num_blocks() {
  return get_num_blocks_x() * get_num_blocks_y() * get_num_blocks_z();
}

/// Returns the 'x' dimension of the current CUDA block's id.
LIBC_INLINE uint32_t get_block_id_x() { return __nvvm_read_ptx_sreg_ctaid_x(); }

/// Returns the 'y' dimension of the current CUDA block's id.
LIBC_INLINE uint32_t get_block_id_y() { return __nvvm_read_ptx_sreg_ctaid_y(); }

/// Returns the 'z' dimension of the current CUDA block's id.
LIBC_INLINE uint32_t get_block_id_z() { return __nvvm_read_ptx_sreg_ctaid_z(); }

/// Returns the absolute id of the CUDA block.
LIBC_INLINE uint64_t get_block_id() {
  return get_block_id_x() + get_num_blocks_x() * get_block_id_y() +
         get_num_blocks_x() * get_num_blocks_y() * get_block_id_z();
}

/// Returns the number of CUDA threads in the 'x' dimension.
LIBC_INLINE uint32_t get_num_threads_x() {
  return __nvvm_read_ptx_sreg_ntid_x();
}

/// Returns the number of CUDA threads in the 'y' dimension.
LIBC_INLINE uint32_t get_num_threads_y() {
  return __nvvm_read_ptx_sreg_ntid_y();
}

/// Returns the number of CUDA threads in the 'z' dimension.
LIBC_INLINE uint32_t get_num_threads_z() {
  return __nvvm_read_ptx_sreg_ntid_z();
}

/// Returns the total number of threads in the block.
LIBC_INLINE uint64_t get_num_threads() {
  return get_num_threads_x() * get_num_threads_y() * get_num_threads_z();
}

/// Returns the 'x' dimension id of the thread in the current CUDA block.
LIBC_INLINE uint32_t get_thread_id_x() { return __nvvm_read_ptx_sreg_tid_x(); }

/// Returns the 'y' dimension id of the thread in the current CUDA block.
LIBC_INLINE uint32_t get_thread_id_y() { return __nvvm_read_ptx_sreg_tid_y(); }

/// Returns the 'z' dimension id of the thread in the current CUDA block.
LIBC_INLINE uint32_t get_thread_id_z() { return __nvvm_read_ptx_sreg_tid_z(); }

/// Returns the absolute id of the thread in the current CUDA block.
LIBC_INLINE uint64_t get_thread_id() {
  return get_thread_id_x() + get_num_threads_x() * get_thread_id_y() +
         get_num_threads_x() * get_num_threads_y() * get_thread_id_z();
}

/// Returns the size of a CUDA warp.
LIBC_INLINE uint32_t get_lane_size() { return LANE_SIZE; }

/// Returns the id of the thread inside of a CUDA warp executing together.
[[clang::convergent]] LIBC_INLINE uint32_t get_lane_id() {
  return get_thread_id() & (get_lane_size() - 1);
}

/// Returns the bit-mask of active threads in the current warp.
[[clang::convergent]] LIBC_INLINE uint64_t get_lane_mask() {
  uint32_t mask;
  LIBC_INLINE_ASM("activemask.b32 %0;" : "=r"(mask));
  return mask;
}

/// Copies the value from the first active thread in the warp to the rest.
[[clang::convergent]] LIBC_INLINE uint32_t broadcast_value(uint64_t lane_mask,
                                                           uint32_t x) {
  uint32_t mask = static_cast<uint32_t>(lane_mask);
  uint32_t id = __builtin_ffs(mask) - 1;
#if __CUDA_ARCH__ >= 600
  return __nvvm_shfl_sync_idx_i32(mask, x, id, get_lane_size() - 1);
#else
  return __nvvm_shfl_idx_i32(x, id, get_lane_size() - 1);
#endif
}

/// Returns a bitmask of threads in the current lane for which \p x is true.
[[clang::convergent]] LIBC_INLINE uint64_t ballot(uint64_t lane_mask, bool x) {
  uint32_t mask = static_cast<uint32_t>(lane_mask);
#if __CUDA_ARCH__ >= 600
  return __nvvm_vote_ballot_sync(mask, x);
#else
  return mask & __nvvm_vote_ballot(x);
#endif
}
/// Waits for all the threads in the block to converge and issues a fence.
[[clang::convergent]] LIBC_INLINE void sync_threads() { __syncthreads(); }

/// Waits for all threads in the warp to reconverge for independent scheduling.
[[clang::convergent]] LIBC_INLINE void sync_lane(uint64_t mask) {
  __nvvm_bar_warp_sync(static_cast<uint32_t>(mask));
}

/// Returns the current value of the GPU's processor clock.
LIBC_INLINE uint64_t processor_clock() {
  uint64_t timestamp;
  LIBC_INLINE_ASM("mov.u64  %0, %%clock64;" : "=l"(timestamp));
  return timestamp;
}

/// Returns a global fixed-frequency timer at nanosecond frequency.
LIBC_INLINE uint64_t fixed_frequency_clock() {
  uint64_t nsecs;
  LIBC_INLINE_ASM("mov.u64  %0, %%globaltimer;" : "=l"(nsecs));
  return nsecs;
}

/// Terminates execution of the calling thread.
[[noreturn]] LIBC_INLINE void end_program() {
  LIBC_INLINE_ASM("exit;" ::: "memory");
  __builtin_unreachable();
}

} // namespace gpu
} // namespace LIBC_NAMESPACE

#endif
