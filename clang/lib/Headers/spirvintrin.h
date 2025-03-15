//===-- spirvintrin.h - SPIRV intrinsic functions ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __SPIRVINTRIN_H
#define __SPIRVINTRIN_H

#ifndef __SPIRV64__
// 32 bit SPIRV is currently a stretch goal
#error "This file is intended for SPIRV64 targets or offloading to SPIRV64"
#endif

#ifndef __GPUINTRIN_H
#error "Never use <spirvintrin.h> directly; include <gpuintrin.h> instead"
#endif

// This is the skeleton of the spirv implementation for gpuintrin
// Address spaces and kernel attribute are not yet implemented
// The target-specific functions are declarations waiting for clang support

#if defined(_OPENMP)
#error "Openmp is not yet available on spirv though gpuintrin header"
#endif

// Type aliases to the address spaces used by the SPIRV backend.
#define __gpu_private
#define __gpu_constant
#define __gpu_local
#define __gpu_global
#define __gpu_generic

// Attribute to declare a function as a kernel.
#define __gpu_kernel

// Returns the number of workgroups in the 'x' dimension of the grid.
_DEFAULT_FN_ATTRS uint32_t __gpu_num_blocks_x(void);

// Returns the number of workgroups in the 'y' dimension of the grid.
_DEFAULT_FN_ATTRS uint32_t __gpu_num_blocks_y(void);

// Returns the number of workgroups in the 'z' dimension of the grid.
_DEFAULT_FN_ATTRS uint32_t __gpu_num_blocks_z(void);

// Returns the 'x' dimension of the current AMD workgroup's id.
_DEFAULT_FN_ATTRS uint32_t __gpu_block_id_x(void);

// Returns the 'y' dimension of the current AMD workgroup's id.
_DEFAULT_FN_ATTRS uint32_t __gpu_block_id_y(void);

// Returns the 'z' dimension of the current AMD workgroup's id.
_DEFAULT_FN_ATTRS uint32_t __gpu_block_id_z(void);

// Returns the number of workitems in the 'x' dimension.
_DEFAULT_FN_ATTRS uint32_t __gpu_num_threads_x(void);

// Returns the number of workitems in the 'y' dimension.
_DEFAULT_FN_ATTRS uint32_t __gpu_num_threads_y(void);

// Returns the number of workitems in the 'z' dimension.
_DEFAULT_FN_ATTRS uint32_t __gpu_num_threads_z(void);

// Returns the 'x' dimension id of the workitem in the current workgroup.
_DEFAULT_FN_ATTRS uint32_t __gpu_thread_id_x(void);

// Returns the 'y' dimension id of the workitem in the current workgroup.
_DEFAULT_FN_ATTRS uint32_t __gpu_thread_id_y(void);

// Returns the 'z' dimension id of the workitem in the current workgroup.
_DEFAULT_FN_ATTRS uint32_t __gpu_thread_id_z(void);

// Returns the size of the wave.
_DEFAULT_FN_ATTRS uint32_t __gpu_num_lanes(void);

// Returns the id of the thread inside of a wave executing together.
_DEFAULT_FN_ATTRS uint32_t __gpu_lane_id(void);

// Returns the bit-mask of active threads in the current wave.
_DEFAULT_FN_ATTRS uint64_t __gpu_lane_mask(void);

// Copies the value from the first active thread in the wave to the rest.
_DEFAULT_FN_ATTRS uint32_t __gpu_read_first_lane_u32(uint64_t __lane_mask,
                                                     uint32_t __x);

// Returns a bitmask of threads in the current lane for which \p x is true.
_DEFAULT_FN_ATTRS uint64_t __gpu_ballot(uint64_t __lane_mask, bool __x);

// Waits for all the threads in the block to converge and issues a fence.
_DEFAULT_FN_ATTRS void __gpu_sync_threads(void);

// Wait for all threads in the wave to converge
_DEFAULT_FN_ATTRS void __gpu_sync_lane(uint64_t __lane_mask);

// Shuffles the the lanes inside the wave according to the given index.
_DEFAULT_FN_ATTRS uint32_t __gpu_shuffle_idx_u32(uint64_t __lane_mask,
                                                 uint32_t __idx, uint32_t __x,
                                                 uint32_t __width);

// Returns a bitmask marking all lanes that have the same value of __x.
_DEFAULT_FN_ATTRS static __inline__ uint64_t
__gpu_match_any_u32(uint64_t __lane_mask, uint32_t __x) {
  return __gpu_match_any_u32_impl(__lane_mask, __x);
}

// Returns a bitmask marking all lanes that have the same value of __x.
_DEFAULT_FN_ATTRS static __inline__ uint64_t
__gpu_match_any_u64(uint64_t __lane_mask, uint64_t __x) {
  return __gpu_match_any_u64_impl(__lane_mask, __x);
}

// Returns the current lane mask if every lane contains __x.
_DEFAULT_FN_ATTRS static __inline__ uint64_t
__gpu_match_all_u32(uint64_t __lane_mask, uint32_t __x) {
  return __gpu_match_all_u32_impl(__lane_mask, __x);
}

// Returns the current lane mask if every lane contains __x.
_DEFAULT_FN_ATTRS static __inline__ uint64_t
__gpu_match_all_u64(uint64_t __lane_mask, uint64_t __x) {
  return __gpu_match_all_u64_impl(__lane_mask, __x);
}

// Terminates execution of the associated wave.
_DEFAULT_FN_ATTRS [[noreturn]] void __gpu_exit(void);

// Suspend the thread briefly to assist the scheduler during busy loops.
_DEFAULT_FN_ATTRS void __gpu_thread_suspend(void);

#endif // __SPIRVINTRIN_H
