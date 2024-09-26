//===-- nvptxintrin.h - NVPTX intrinsic functions -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __NVPTXINTRIN_H
#define __NVPTXINTRIN_H

#ifndef __NVPTX__
#error "This file is intended for NVPTX targets or offloading to NVPTX
#endif

#include <stdbool.h>
#include <stdint.h>

#if defined(__HIP__) || defined(__CUDA__)
#define _DEFAULT_ATTRS __attribute__((device)) __attribute__((always_inline))
#else
#define _DEFAULT_ATTRS __attribute__((always_inline))
#endif

#pragma omp begin declare target device_type(nohost)
#pragma omp begin declare variant match(device = {arch(nvptx64)})

// Type aliases to the address spaces used by the NVPTX backend.
#define _private __attribute__((opencl_private))
#define _constant __attribute__((opencl_constant))
#define _local __attribute__((opencl_local))
#define _global __attribute__((opencl_global))

// Attribute to declare a function as a kernel.
#define _kernel __attribute__((nvptx_kernel))

// Returns the number of CUDA blocks in the 'x' dimension.
_DEFAULT_ATTRS static inline uint32_t _get_num_blocks_x() {
  return __nvvm_read_ptx_sreg_nctaid_x();
}

// Returns the number of CUDA blocks in the 'y' dimension.
_DEFAULT_ATTRS static inline uint32_t _get_num_blocks_y() {
  return __nvvm_read_ptx_sreg_nctaid_y();
}

// Returns the number of CUDA blocks in the 'z' dimension.
_DEFAULT_ATTRS static inline uint32_t _get_num_blocks_z() {
  return __nvvm_read_ptx_sreg_nctaid_z();
}

// Returns the total number of CUDA blocks.
_DEFAULT_ATTRS static inline uint64_t _get_num_blocks() {
  return _get_num_blocks_x() * _get_num_blocks_y() * _get_num_blocks_z();
}

// Returns the 'x' dimension of the current CUDA block's id.
_DEFAULT_ATTRS static inline uint32_t _get_block_id_x() {
  return __nvvm_read_ptx_sreg_ctaid_x();
}

// Returns the 'y' dimension of the current CUDA block's id.
_DEFAULT_ATTRS static inline uint32_t _get_block_id_y() {
  return __nvvm_read_ptx_sreg_ctaid_y();
}

// Returns the 'z' dimension of the current CUDA block's id.
_DEFAULT_ATTRS static inline uint32_t _get_block_id_z() {
  return __nvvm_read_ptx_sreg_ctaid_z();
}

// Returns the absolute id of the CUDA block.
_DEFAULT_ATTRS static inline uint64_t _get_block_id() {
  return _get_block_id_x() + _get_num_blocks_x() * _get_block_id_y() +
         _get_num_blocks_x() * _get_num_blocks_y() * _get_block_id_z();
}

// Returns the number of CUDA threads in the 'x' dimension.
_DEFAULT_ATTRS static inline uint32_t _get_num_threads_x() {
  return __nvvm_read_ptx_sreg_ntid_x();
}

// Returns the number of CUDA threads in the 'y' dimension.
_DEFAULT_ATTRS static inline uint32_t _get_num_threads_y() {
  return __nvvm_read_ptx_sreg_ntid_y();
}

// Returns the number of CUDA threads in the 'z' dimension.
_DEFAULT_ATTRS static inline uint32_t _get_num_threads_z() {
  return __nvvm_read_ptx_sreg_ntid_z();
}

// Returns the total number of threads in the block.
_DEFAULT_ATTRS static inline uint64_t _get_num_threads() {
  return _get_num_threads_x() * _get_num_threads_y() * _get_num_threads_z();
}

// Returns the 'x' dimension id of the thread in the current CUDA block.
_DEFAULT_ATTRS static inline uint32_t _get_thread_id_x() {
  return __nvvm_read_ptx_sreg_tid_x();
}

// Returns the 'y' dimension id of the thread in the current CUDA block.
_DEFAULT_ATTRS static inline uint32_t _get_thread_id_y() {
  return __nvvm_read_ptx_sreg_tid_y();
}

// Returns the 'z' dimension id of the thread in the current CUDA block.
_DEFAULT_ATTRS static inline uint32_t _get_thread_id_z() {
  return __nvvm_read_ptx_sreg_tid_z();
}

// Returns the absolute id of the thread in the current CUDA block.
_DEFAULT_ATTRS static inline uint64_t _get_thread_id() {
  return _get_thread_id_x() + _get_num_threads_x() * _get_thread_id_y() +
         _get_num_threads_x() * _get_num_threads_y() * _get_thread_id_z();
}

// Returns the size of a CUDA warp, always 32 on NVIDIA hardware.
_DEFAULT_ATTRS static inline uint32_t _get_lane_size() { return 32; }

// Returns the id of the thread inside of a CUDA warp executing together.
_DEFAULT_ATTRS [[clang::convergent]] static inline uint32_t _get_lane_id() {
  return __nvvm_read_ptx_sreg_laneid();
}

// Returns the bit-mask of active threads in the current warp.
_DEFAULT_ATTRS [[clang::convergent]] static inline uint64_t _get_lane_mask() {
  return __nvvm_activemask();
}

// Copies the value from the first active thread in the warp to the rest.
_DEFAULT_ATTRS [[clang::convergent]] static inline uint32_t
_broadcast_value(uint64_t lane_mask, uint32_t x) {
  uint32_t mask = static_cast<uint32_t>(lane_mask);
  uint32_t id = __builtin_ffs(mask) - 1;
  return __nvvm_shfl_sync_idx_i32(mask, x, id, _get_lane_size() - 1);
}

// Returns a bitmask of threads in the current lane for which \p x is true.
_DEFAULT_ATTRS [[clang::convergent]] static inline uint64_t
_ballot(uint64_t lane_mask, bool x) {
  uint32_t mask = static_cast<uint32_t>(lane_mask);
  return __nvvm_vote_ballot_sync(mask, x);
}

// Waits for all the threads in the block to converge and issues a fence.
_DEFAULT_ATTRS [[clang::convergent]] static inline void _sync_threads() {
  __syncthreads();
}

// Waits for all threads in the warp to reconverge for independent scheduling.
_DEFAULT_ATTRS [[clang::convergent]] static inline void
_sync_lane(uint64_t mask) {
  __nvvm_bar_warp_sync(static_cast<uint32_t>(mask));
}

// Shuffles the the lanes inside the warp according to the given index.
_DEFAULT_ATTRS [[clang::convergent]] static inline uint32_t
_shuffle(uint64_t lane_mask, uint32_t idx, uint32_t x) {
  uint32_t mask = static_cast<uint32_t>(lane_mask);
  uint32_t bitmask = (mask >> idx) & 1;
  return -bitmask &
         __nvvm_shfl_sync_idx_i32(mask, x, idx, _get_lane_size() - 1);
}

// Returns the current value of the GPU's processor clock.
_DEFAULT_ATTRS static inline uint64_t _processor_clock() {
  return __builtin_readcyclecounter();
}

// Returns a global fixed-frequency timer at nanosecond frequency.
_DEFAULT_ATTRS static inline uint64_t _fixed_frequency_clock() {
  return __builtin_readsteadycounter();
}

// Terminates execution of the calling thread.
_DEFAULT_ATTRS [[noreturn]] static inline void _end_program() { __nvvm_exit(); }

#pragma omp end declare variant
#pragma omp end declare target
#undef _DEFAULT_ATTRS

#endif // __NVPTXINTRIN_H
