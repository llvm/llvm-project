//===-- spirvintrin.h - SPIR-V intrinsic functions ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __SPIRVINTRIN_H
#define __SPIRVINTRIN_H

#ifndef __SPIRV__
#error "This file is intended for SPIR-V targets or offloading to SPIR-V"
#endif

#ifndef __GPUINTRIN_H
#error "Never use <spirvintrin.h> directly; include <gpuintrin.h> instead"
#endif

_Pragma("omp begin declare target device_type(nohost)");
_Pragma("omp begin declare variant match(device = {arch(spirv64)})");

// Type aliases to the address spaces used by the SPIR-V backend.
#define __gpu_private __attribute__((address_space(0)))
#define __gpu_constant __attribute__((address_space(2)))
#define __gpu_local __attribute__((address_space(3)))
#define __gpu_global __attribute__((address_space(1)))
#define __gpu_generic __attribute__((address_space(4)))

// Returns the number of workgroups in the 'x' dimension of the grid.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_num_blocks_x(void) {
  return __builtin_spirv_num_workgroups(0);
}

// Returns the number of workgroups in the 'y' dimension of the grid.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_num_blocks_y(void) {
  return __builtin_spirv_num_workgroups(1);
}

// Returns the number of workgroups in the 'z' dimension of the grid.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_num_blocks_z(void) {
  return __builtin_spirv_num_workgroups(2);
}

// Returns the 'x' dimension of the current workgroup's id.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_block_id_x(void) {
  return __builtin_spirv_workgroup_id(0);
}

// Returns the 'y' dimension of the current workgroup's id.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_block_id_y(void) {
  return __builtin_spirv_workgroup_id(1);
}

// Returns the 'z' dimension of the current workgroup's id.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_block_id_z(void) {
  return __builtin_spirv_workgroup_id(2);
}

// Returns the number of workitems in the 'x' dimension.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_num_threads_x(void) {
  return __builtin_spirv_workgroup_size(0);
}

// Returns the number of workitems in the 'y' dimension.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_num_threads_y(void) {
  return __builtin_spirv_workgroup_size(1);
}

// Returns the number of workitems in the 'z' dimension.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_num_threads_z(void) {
  return __builtin_spirv_workgroup_size(2);
}

// Returns the 'x' dimension id of the workitem in the current workgroup.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_thread_id_x(void) {
  return __builtin_spirv_local_invocation_id(0);
}

// Returns the 'y' dimension id of the workitem in the current workgroup.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_thread_id_y(void) {
  return __builtin_spirv_local_invocation_id(1);
}

// Returns the 'z' dimension id of the workitem in the current workgroup.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_thread_id_z(void) {
  return __builtin_spirv_local_invocation_id(2);
}

// Returns the size of an wavefront, either 32 or 64 depending on hardware
// and compilation options.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_num_lanes(void) {
  return __builtin_spirv_subgroup_size();
}

// Returns the id of the thread inside of an wavefront executing together.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_lane_id(void) {
  return __builtin_spirv_subgroup_local_invocation_id();
}

// Returns the bit-mask of active threads in the current wavefront. This
// implementation is incorrect if the target uses more than 64 lanes.
_DEFAULT_FN_ATTRS static __inline__ uint64_t __gpu_lane_mask(void) {
  uint32_t [[clang::ext_vector_type(4)]] __mask =
      __builtin_spirv_subgroup_ballot(1);
  return __builtin_bit_cast(uint64_t,
                            __builtin_shufflevector(__mask, __mask, 0, 1));
}

// Copies the value from the first active thread in the wavefront to the rest.
_DEFAULT_FN_ATTRS static __inline__ uint32_t
__gpu_read_first_lane_u32(uint64_t __lane_mask, uint32_t __x) {
  return __builtin_spirv_subgroup_shuffle(__x,
                                          __builtin_ctzg(__gpu_lane_mask()));
}

// Returns a bitmask of threads in the current lane for which \p x is true. This
// implementation is incorrect if the target uses more than 64 lanes.
_DEFAULT_FN_ATTRS static __inline__ uint64_t __gpu_ballot(uint64_t __lane_mask,
                                                          bool __x) {
  // The lane_mask & gives the nvptx semantics when lane_mask is a subset of
  // the active threads.
  uint32_t [[clang::ext_vector_type(4)]] __mask =
      __builtin_spirv_subgroup_ballot(__x);
  return __lane_mask & __builtin_bit_cast(uint64_t, __builtin_shufflevector(
                                                        __mask, __mask, 0, 1));
}

// Waits for all the threads in the block to converge and issues a fence.
_DEFAULT_FN_ATTRS static __inline__ void __gpu_sync_threads(void) {
  __builtin_spirv_group_barrier();
}

// Wait for all threads in the wavefront to converge, this is a noop on SPIR-V.
_DEFAULT_FN_ATTRS static __inline__ void __gpu_sync_lane(uint64_t __lane_mask) {
}

// Shuffles the the lanes inside the wavefront according to the given index.
_DEFAULT_FN_ATTRS static __inline__ uint32_t
__gpu_shuffle_idx_u32(uint64_t __lane_mask, uint32_t __idx, uint32_t __x,
                      uint32_t __width) {
  uint32_t __lane = __idx + (__gpu_lane_id() & ~(__width - 1));
  return __builtin_spirv_subgroup_shuffle(__x, __lane);
}

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

// SPIR-V does not expose this, always return false.
_DEFAULT_FN_ATTRS static __inline__ bool __gpu_is_ptr_local(void *ptr) {
  return 0;
}

// SPIR-V does not expose this, always return false.
_DEFAULT_FN_ATTRS static __inline__ bool __gpu_is_ptr_private(void *ptr) {
  return 0;
}

// SPIR-V only supports 'OpTerminateInvocation' in fragment shaders.
_DEFAULT_FN_ATTRS [[noreturn]] static __inline__ void __gpu_exit(void) {
  __builtin_trap();
}

// This is a no-op as SPIR-V does not support it.
_DEFAULT_FN_ATTRS static __inline__ void __gpu_thread_suspend(void) {}

_Pragma("omp end declare variant");
_Pragma("omp end declare target");

#endif // __SPIRVINTRIN_H
