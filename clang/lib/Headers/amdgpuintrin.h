//===-- amdgpuintrin.h - AMDPGU intrinsic functions -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __AMDGPUINTRIN_H
#define __AMDGPUINTRIN_H

#ifndef __AMDGPU__
#error "This file is intended for AMDGPU targets or offloading to AMDGPU
#endif

#include <stdbool.h>
#include <stdint.h>

#if defined(__HIP__) || defined(__CUDA__)
#define _DEFAULT_ATTRS __attribute__((device)) __attribute__((always_inline))
#else
#define _DEFAULT_ATTRS __attribute__((always_inline))
#endif

#pragma omp begin declare target device_type(nohost)
#pragma omp begin declare variant match(device = {arch(amdgcn)})

// Type aliases to the address spaces used by the AMDGPU backend.
#define _private __attribute__((opencl_private))
#define _constant __attribute__((opencl_constant))
#define _local __attribute__((opencl_local))
#define _global __attribute__((opencl_global))

// Attribute to declare a function as a kernel.
#define _kernel __attribute__((amdgpu_kernel, visibility("protected")))

// Returns the number of workgroups in the 'x' dimension of the grid.
_DEFAULT_ATTRS static inline uint32_t _get_num_blocks_x() {
  return __builtin_amdgcn_grid_size_x() / __builtin_amdgcn_workgroup_size_x();
}

// Returns the number of workgroups in the 'y' dimension of the grid.
_DEFAULT_ATTRS static inline uint32_t _get_num_blocks_y() {
  return __builtin_amdgcn_grid_size_y() / __builtin_amdgcn_workgroup_size_y();
}

// Returns the number of workgroups in the 'z' dimension of the grid.
_DEFAULT_ATTRS static inline uint32_t _get_num_blocks_z() {
  return __builtin_amdgcn_grid_size_z() / __builtin_amdgcn_workgroup_size_z();
}

// Returns the total number of workgruops in the grid.
_DEFAULT_ATTRS static inline uint64_t _get_num_blocks() {
  return _get_num_blocks_x() * _get_num_blocks_y() * _get_num_blocks_z();
}

// Returns the 'x' dimension of the current AMD workgroup's id.
_DEFAULT_ATTRS static inline uint32_t _get_block_id_x() {
  return __builtin_amdgcn_workgroup_id_x();
}

// Returns the 'y' dimension of the current AMD workgroup's id.
_DEFAULT_ATTRS static inline uint32_t _get_block_id_y() {
  return __builtin_amdgcn_workgroup_id_y();
}

// Returns the 'z' dimension of the current AMD workgroup's id.
_DEFAULT_ATTRS static inline uint32_t _get_block_id_z() {
  return __builtin_amdgcn_workgroup_id_z();
}

// Returns the absolute id of the AMD workgroup.
_DEFAULT_ATTRS static inline uint64_t _get_block_id() {
  return _get_block_id_x() + _get_num_blocks_x() * _get_block_id_y() +
         _get_num_blocks_x() * _get_num_blocks_y() * _get_block_id_z();
}

// Returns the number of workitems in the 'x' dimension.
_DEFAULT_ATTRS static inline uint32_t _get_num_threads_x() {
  return __builtin_amdgcn_workgroup_size_x();
}

// Returns the number of workitems in the 'y' dimension.
_DEFAULT_ATTRS static inline uint32_t _get_num_threads_y() {
  return __builtin_amdgcn_workgroup_size_y();
}

// Returns the number of workitems in the 'z' dimension.
_DEFAULT_ATTRS static inline uint32_t _get_num_threads_z() {
  return __builtin_amdgcn_workgroup_size_z();
}

// Returns the total number of workitems in the workgroup.
_DEFAULT_ATTRS static inline uint64_t _get_num_threads() {
  return _get_num_threads_x() * _get_num_threads_y() * _get_num_threads_z();
}

// Returns the 'x' dimension id of the workitem in the current AMD workgroup.
_DEFAULT_ATTRS static inline uint32_t _get_thread_id_x() {
  return __builtin_amdgcn_workitem_id_x();
}

// Returns the 'y' dimension id of the workitem in the current AMD workgroup.
_DEFAULT_ATTRS static inline uint32_t _get_thread_id_y() {
  return __builtin_amdgcn_workitem_id_y();
}

// Returns the 'z' dimension id of the workitem in the current AMD workgroup.
_DEFAULT_ATTRS static inline uint32_t _get_thread_id_z() {
  return __builtin_amdgcn_workitem_id_z();
}

// Returns the absolute id of the thread in the current AMD workgroup.
_DEFAULT_ATTRS static inline uint64_t _get_thread_id() {
  return _get_thread_id_x() + _get_num_threads_x() * _get_thread_id_y() +
         _get_num_threads_x() * _get_num_threads_y() * _get_thread_id_z();
}

// Returns the size of an AMD wavefront, either 32 or 64 depending on hardware
// and compilation options.
_DEFAULT_ATTRS static inline uint32_t _get_lane_size() {
  return __builtin_amdgcn_wavefrontsize();
}

// Returns the id of the thread inside of an AMD wavefront executing together.
_DEFAULT_ATTRS [[clang::convergent]] static inline uint32_t _get_lane_id() {
  return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}

// Returns the bit-mask of active threads in the current wavefront.
_DEFAULT_ATTRS [[clang::convergent]] static inline uint64_t _get_lane_mask() {
  return __builtin_amdgcn_read_exec();
}

// Copies the value from the first active thread in the wavefront to the rest.
_DEFAULT_ATTRS [[clang::convergent]] static inline uint32_t
_broadcast_value(uint64_t, uint32_t x) {
  return __builtin_amdgcn_readfirstlane(x);
}

// Returns a bitmask of threads in the current lane for which \p x is true.
_DEFAULT_ATTRS [[clang::convergent]] static inline uint64_t
_ballot(uint64_t lane_mask, bool x) {
  // The lane_mask & gives the nvptx semantics when lane_mask is a subset of
  // the active threads
  return lane_mask & __builtin_amdgcn_ballot_w64(x);
}

// Waits for all the threads in the block to converge and issues a fence.
_DEFAULT_ATTRS [[clang::convergent]] static inline void _sync_threads() {
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}

// Wait for all threads in the wavefront to converge, this is a noop on AMDGPU.
_DEFAULT_ATTRS [[clang::convergent]] static inline void _sync_lane(uint64_t) {
  __builtin_amdgcn_wave_barrier();
}

// Shuffles the the lanes inside the wavefront according to the given index.
_DEFAULT_ATTRS [[clang::convergent]] static inline uint32_t
_shuffle(uint64_t, uint32_t idx, uint32_t x) {
  return __builtin_amdgcn_ds_bpermute(idx << 2, x);
}

// Returns the current value of the GPU's processor clock.
// NOTE: The RDNA3 and RDNA2 architectures use a 20-bit cycle counter.
_DEFAULT_ATTRS static inline uint64_t _processor_clock() {
  return __builtin_readcyclecounter();
}

// Returns a fixed-frequency timestamp. The actual frequency is dependent on
// the card and can only be queried via the driver.
_DEFAULT_ATTRS static inline uint64_t _fixed_frequency_clock() {
  return __builtin_readsteadycounter();
}

// Terminates execution of the associated wavefront.
_DEFAULT_ATTRS [[noreturn]] static inline void _end_program() {
  __builtin_amdgcn_endpgm();
}

#pragma omp end declare variant
#pragma omp end declare target
#undef _DEFAULT_ATTRS

#endif // __AMDGPUINTRIN_H
