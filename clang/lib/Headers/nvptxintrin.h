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
#error "This file is intended for NVPTX targets or offloading to NVPTX"
#endif

#include <stdbool.h>
#include <stdint.h>

#if defined(__HIP__) || defined(__CUDA__)
#define _DEFAULT_ATTRS __attribute__((device))
#elif !defined(_DEFAULT_ATTRS)
#define _DEFAULT_ATTRS
#endif

#pragma omp begin declare target device_type(nohost)
#pragma omp begin declare variant match(device = {arch(nvptx64)})

// Type aliases to the address spaces used by the NVPTX backend.
#define _Private __attribute__((opencl_private))
#define _Constant __attribute__((opencl_constant))
#define _Local __attribute__((opencl_local))
#define _Global __attribute__((opencl_global))

// Attribute to declare a function as a kernel.
#define _Kernel __attribute__((nvptx_kernel, visibility("protected")))

// Returns the number of CUDA blocks in the 'x' dimension.
_DEFAULT_ATTRS static inline uint32_t __gpu_num_blocks_x() {
  return __nvvm_read_ptx_sreg_nctaid_x();
}

// Returns the number of CUDA blocks in the 'y' dimension.
_DEFAULT_ATTRS static inline uint32_t __gpu_num_blocks_y() {
  return __nvvm_read_ptx_sreg_nctaid_y();
}

// Returns the number of CUDA blocks in the 'z' dimension.
_DEFAULT_ATTRS static inline uint32_t __gpu_num_blocks_z() {
  return __nvvm_read_ptx_sreg_nctaid_z();
}

// Returns the 'x' dimension of the current CUDA block's id.
_DEFAULT_ATTRS static inline uint32_t __gpu_block_id_x() {
  return __nvvm_read_ptx_sreg_ctaid_x();
}

// Returns the 'y' dimension of the current CUDA block's id.
_DEFAULT_ATTRS static inline uint32_t __gpu_block_id_y() {
  return __nvvm_read_ptx_sreg_ctaid_y();
}

// Returns the 'z' dimension of the current CUDA block's id.
_DEFAULT_ATTRS static inline uint32_t __gpu_block_id_z() {
  return __nvvm_read_ptx_sreg_ctaid_z();
}

// Returns the number of CUDA threads in the 'x' dimension.
_DEFAULT_ATTRS static inline uint32_t __gpu_num_threads_x() {
  return __nvvm_read_ptx_sreg_ntid_x();
}

// Returns the number of CUDA threads in the 'y' dimension.
_DEFAULT_ATTRS static inline uint32_t __gpu_num_threads_y() {
  return __nvvm_read_ptx_sreg_ntid_y();
}

// Returns the number of CUDA threads in the 'z' dimension.
_DEFAULT_ATTRS static inline uint32_t __gpu_num_threads_z() {
  return __nvvm_read_ptx_sreg_ntid_z();
}

// Returns the 'x' dimension id of the thread in the current CUDA block.
_DEFAULT_ATTRS static inline uint32_t __gpu_thread_id_x() {
  return __nvvm_read_ptx_sreg_tid_x();
}

// Returns the 'y' dimension id of the thread in the current CUDA block.
_DEFAULT_ATTRS static inline uint32_t __gpu_thread_id_y() {
  return __nvvm_read_ptx_sreg_tid_y();
}

// Returns the 'z' dimension id of the thread in the current CUDA block.
_DEFAULT_ATTRS static inline uint32_t __gpu_thread_id_z() {
  return __nvvm_read_ptx_sreg_tid_z();
}

// Returns the size of a CUDA warp, always 32 on NVIDIA hardware.
_DEFAULT_ATTRS static inline uint32_t __gpu_num_lanes() {
  return __nvvm_read_ptx_sreg_warpsize();
}

// Returns the id of the thread inside of a CUDA warp executing together.
_DEFAULT_ATTRS [[clang::convergent]] static inline uint32_t __gpu_lane_id() {
  return __nvvm_read_ptx_sreg_laneid();
}

// Returns the bit-mask of active threads in the current warp.
_DEFAULT_ATTRS [[clang::convergent]] static inline uint64_t __gpu_lane_mask() {
  return __nvvm_activemask();
}

// Copies the value from the first active thread in the warp to the rest.
_DEFAULT_ATTRS [[clang::convergent]] static inline uint32_t
__gpu_broadcast(uint64_t __lane_mask, uint32_t __x) {
  uint32_t __mask = (uint32_t)__lane_mask;
  uint32_t __id = __builtin_ffs(__mask) - 1;
  return __nvvm_shfl_sync_idx_i32(__mask, __x, __id, __gpu_num_lanes() - 1);
}

// Returns a bitmask of threads in the current lane for which \p x is true.
_DEFAULT_ATTRS [[clang::convergent]] static inline uint64_t
__gpu_ballot(uint64_t __lane_mask, bool __x) {
  uint32_t __mask = (uint32_t)__lane_mask;
  return __nvvm_vote_ballot_sync(__mask, __x);
}

// Waits for all the threads in the block to converge and issues a fence.
_DEFAULT_ATTRS [[clang::convergent]] static inline void __gpu_sync_threads() {
  __syncthreads();
}

// Waits for all threads in the warp to reconverge for independent scheduling.
_DEFAULT_ATTRS [[clang::convergent]] static inline void
__gpu_sync_lane(uint64_t __lane_mask) {
  __nvvm_bar_warp_sync((uint32_t)__lane_mask);
}

// Shuffles the the lanes inside the warp according to the given index.
_DEFAULT_ATTRS [[clang::convergent]] static inline uint32_t
__gpu_shuffle_idx(uint64_t __lane_mask, uint32_t __idx, uint32_t __x) {
  uint32_t __mask = (uint32_t)__lane_mask;
  uint32_t __bitmask = (__mask >> __idx) & 1u;
  return -__bitmask &
         __nvvm_shfl_sync_idx_i32(__mask, __x, __idx, __gpu_num_lanes() - 1u);
}

// Terminates execution of the calling thread.
_DEFAULT_ATTRS [[noreturn]] static inline void __gpu_exit() { __nvvm_exit(); }

#pragma omp end declare variant
#pragma omp end declare target

#endif // __NVPTXINTRIN_H
