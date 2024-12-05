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

#include <stdint.h>

#if !defined(__cplusplus)
_Pragma("push_macro(\"bool\")");
#define bool _Bool
#endif

_Pragma("omp begin declare target device_type(nohost)");
_Pragma("omp begin declare variant match(device = {arch(nvptx64)})");

// Type aliases to the address spaces used by the NVPTX backend.
#define __gpu_private __attribute__((address_space(5)))
#define __gpu_constant __attribute__((address_space(4)))
#define __gpu_local __attribute__((address_space(3)))
#define __gpu_global __attribute__((address_space(1)))
#define __gpu_generic __attribute__((address_space(0)))

// Attribute to declare a function as a kernel.
#define __gpu_kernel __attribute__((nvptx_kernel, visibility("protected")))

// Returns the number of CUDA blocks in the 'x' dimension.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_num_blocks_x(void) {
  return __nvvm_read_ptx_sreg_nctaid_x();
}

// Returns the number of CUDA blocks in the 'y' dimension.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_num_blocks_y(void) {
  return __nvvm_read_ptx_sreg_nctaid_y();
}

// Returns the number of CUDA blocks in the 'z' dimension.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_num_blocks_z(void) {
  return __nvvm_read_ptx_sreg_nctaid_z();
}

// Returns the 'x' dimension of the current CUDA block's id.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_block_id_x(void) {
  return __nvvm_read_ptx_sreg_ctaid_x();
}

// Returns the 'y' dimension of the current CUDA block's id.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_block_id_y(void) {
  return __nvvm_read_ptx_sreg_ctaid_y();
}

// Returns the 'z' dimension of the current CUDA block's id.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_block_id_z(void) {
  return __nvvm_read_ptx_sreg_ctaid_z();
}

// Returns the number of CUDA threads in the 'x' dimension.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_num_threads_x(void) {
  return __nvvm_read_ptx_sreg_ntid_x();
}

// Returns the number of CUDA threads in the 'y' dimension.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_num_threads_y(void) {
  return __nvvm_read_ptx_sreg_ntid_y();
}

// Returns the number of CUDA threads in the 'z' dimension.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_num_threads_z(void) {
  return __nvvm_read_ptx_sreg_ntid_z();
}

// Returns the 'x' dimension id of the thread in the current CUDA block.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_thread_id_x(void) {
  return __nvvm_read_ptx_sreg_tid_x();
}

// Returns the 'y' dimension id of the thread in the current CUDA block.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_thread_id_y(void) {
  return __nvvm_read_ptx_sreg_tid_y();
}

// Returns the 'z' dimension id of the thread in the current CUDA block.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_thread_id_z(void) {
  return __nvvm_read_ptx_sreg_tid_z();
}

// Returns the size of a CUDA warp, always 32 on NVIDIA hardware.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_num_lanes(void) {
  return __nvvm_read_ptx_sreg_warpsize();
}

// Returns the id of the thread inside of a CUDA warp executing together.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_lane_id(void) {
  return __nvvm_read_ptx_sreg_laneid();
}

// Returns the bit-mask of active threads in the current warp.
_DEFAULT_FN_ATTRS static __inline__ uint64_t __gpu_lane_mask(void) {
  return __nvvm_activemask();
}

// Copies the value from the first active thread in the warp to the rest.
_DEFAULT_FN_ATTRS static __inline__ uint32_t
__gpu_read_first_lane_u32(uint64_t __lane_mask, uint32_t __x) {
  uint32_t __mask = (uint32_t)__lane_mask;
  uint32_t __id = __builtin_ffs(__mask) - 1;
  return __nvvm_shfl_sync_idx_i32(__mask, __x, __id, __gpu_num_lanes() - 1);
}

// Copies the value from the first active thread in the warp to the rest.
_DEFAULT_FN_ATTRS static __inline__ uint64_t
__gpu_read_first_lane_u64(uint64_t __lane_mask, uint64_t __x) {
  uint32_t __hi = (uint32_t)(__x >> 32ull);
  uint32_t __lo = (uint32_t)(__x & 0xFFFFFFFF);
  uint32_t __mask = (uint32_t)__lane_mask;
  uint32_t __id = __builtin_ffs(__mask) - 1;
  return ((uint64_t)__nvvm_shfl_sync_idx_i32(__mask, __hi, __id,
                                             __gpu_num_lanes() - 1)
          << 32ull) |
         ((uint64_t)__nvvm_shfl_sync_idx_i32(__mask, __lo, __id,
                                             __gpu_num_lanes() - 1));
}

// Returns a bitmask of threads in the current lane for which \p x is true.
_DEFAULT_FN_ATTRS static __inline__ uint64_t __gpu_ballot(uint64_t __lane_mask,
                                                          bool __x) {
  uint32_t __mask = (uint32_t)__lane_mask;
  return __nvvm_vote_ballot_sync(__mask, __x);
}

// Waits for all the threads in the block to converge and issues a fence.
_DEFAULT_FN_ATTRS static __inline__ void __gpu_sync_threads(void) {
  __syncthreads();
}

// Waits for all threads in the warp to reconverge for independent scheduling.
_DEFAULT_FN_ATTRS static __inline__ void __gpu_sync_lane(uint64_t __lane_mask) {
  __nvvm_bar_warp_sync((uint32_t)__lane_mask);
}

// Shuffles the the lanes inside the warp according to the given index.
_DEFAULT_FN_ATTRS static __inline__ uint32_t
__gpu_shuffle_idx_u32(uint64_t __lane_mask, uint32_t __idx, uint32_t __x) {
  uint32_t __mask = (uint32_t)__lane_mask;
  uint32_t __bitmask = (__mask >> __idx) & 1u;
  return -__bitmask &
         __nvvm_shfl_sync_idx_i32(__mask, __x, __idx, __gpu_num_lanes() - 1u);
}

// Shuffles the the lanes inside the warp according to the given index.
_DEFAULT_FN_ATTRS static __inline__ uint64_t
__gpu_shuffle_idx_u64(uint64_t __lane_mask, uint32_t __idx, uint64_t __x) {
  uint32_t __hi = (uint32_t)(__x >> 32ull);
  uint32_t __lo = (uint32_t)(__x & 0xFFFFFFFF);
  uint32_t __mask = (uint32_t)__lane_mask;
  uint64_t __bitmask = (__mask >> __idx) & 1u;
  return -__bitmask & ((uint64_t)__nvvm_shfl_sync_idx_i32(
                           __mask, __hi, __idx, __gpu_num_lanes() - 1u)
                       << 32ull) |
         ((uint64_t)__nvvm_shfl_sync_idx_i32(__mask, __lo, __idx,
                                             __gpu_num_lanes() - 1u));
}

// Returns true if the flat pointer points to CUDA 'shared' memory.
_DEFAULT_FN_ATTRS static __inline__ bool __gpu_is_ptr_local(void *ptr) {
  return __nvvm_isspacep_shared(ptr);
}

// Returns true if the flat pointer points to CUDA 'local' memory.
_DEFAULT_FN_ATTRS static __inline__ bool __gpu_is_ptr_private(void *ptr) {
  return __nvvm_isspacep_local(ptr);
}

// Terminates execution of the calling thread.
_DEFAULT_FN_ATTRS [[noreturn]] static __inline__ void __gpu_exit(void) {
  __nvvm_exit();
}

// Suspend the thread briefly to assist the scheduler during busy loops.
_DEFAULT_FN_ATTRS static __inline__ void __gpu_thread_suspend(void) {
  if (__nvvm_reflect("__CUDA_ARCH") >= 700)
    asm("nanosleep.u32 64;" ::: "memory");
}

_Pragma("omp end declare variant");
_Pragma("omp end declare target");

#if !defined(__cplusplus)
_Pragma("pop_macro(\"bool\")");
#endif

#endif // __NVPTXINTRIN_H
