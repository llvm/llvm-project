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
#error "This file is intended for AMDGPU targets or offloading to AMDGPU"
#endif

#include <stdbool.h>
#include <stdint.h>

#if defined(__HIP__) || defined(__CUDA__)
#define _DEFAULT_ATTRS __attribute__((device))
#elif !defined(_DEFAULT_ATTRS)
#define _DEFAULT_ATTRS
#endif

#pragma omp begin declare target device_type(nohost)
#pragma omp begin declare variant match(device = {arch(amdgcn)})

// Type aliases to the address spaces used by the AMDGPU backend.
#define _Private __attribute__((opencl_private))
#define _Constant __attribute__((opencl_constant))
#define _Local __attribute__((opencl_local))
#define _Global __attribute__((opencl_global))

// Attribute to declare a function as a kernel.
#define _Kernel __attribute__((amdgpu_kernel, visibility("protected")))

// Returns the number of workgroups in the 'x' dimension of the grid.
_DEFAULT_ATTRS static inline uint32_t __gpu_num_blocks_x() {
  return __builtin_amdgcn_grid_size_x() / __builtin_amdgcn_workgroup_size_x();
}

// Returns the number of workgroups in the 'y' dimension of the grid.
_DEFAULT_ATTRS static inline uint32_t __gpu_num_blocks_y() {
  return __builtin_amdgcn_grid_size_y() / __builtin_amdgcn_workgroup_size_y();
}

// Returns the number of workgroups in the 'z' dimension of the grid.
_DEFAULT_ATTRS static inline uint32_t __gpu_num_blocks_z() {
  return __builtin_amdgcn_grid_size_z() / __builtin_amdgcn_workgroup_size_z();
}

// Returns the 'x' dimension of the current AMD workgroup's id.
_DEFAULT_ATTRS static inline uint32_t __gpu_block_id_x() {
  return __builtin_amdgcn_workgroup_id_x();
}

// Returns the 'y' dimension of the current AMD workgroup's id.
_DEFAULT_ATTRS static inline uint32_t __gpu_block_id_y() {
  return __builtin_amdgcn_workgroup_id_y();
}

// Returns the 'z' dimension of the current AMD workgroup's id.
_DEFAULT_ATTRS static inline uint32_t __gpu_block_id_z() {
  return __builtin_amdgcn_workgroup_id_z();
}

// Returns the number of workitems in the 'x' dimension.
_DEFAULT_ATTRS static inline uint32_t __gpu_num_threads_x() {
  return __builtin_amdgcn_workgroup_size_x();
}

// Returns the number of workitems in the 'y' dimension.
_DEFAULT_ATTRS static inline uint32_t __gpu_num_threads_y() {
  return __builtin_amdgcn_workgroup_size_y();
}

// Returns the number of workitems in the 'z' dimension.
_DEFAULT_ATTRS static inline uint32_t __gpu_num_threads_z() {
  return __builtin_amdgcn_workgroup_size_z();
}

// Returns the 'x' dimension id of the workitem in the current AMD workgroup.
_DEFAULT_ATTRS static inline uint32_t __gpu_thread_id_x() {
  return __builtin_amdgcn_workitem_id_x();
}

// Returns the 'y' dimension id of the workitem in the current AMD workgroup.
_DEFAULT_ATTRS static inline uint32_t __gpu_thread_id_y() {
  return __builtin_amdgcn_workitem_id_y();
}

// Returns the 'z' dimension id of the workitem in the current AMD workgroup.
_DEFAULT_ATTRS static inline uint32_t __gpu_thread_id_z() {
  return __builtin_amdgcn_workitem_id_z();
}

// Returns the size of an AMD wavefront, either 32 or 64 depending on hardware
// and compilation options.
_DEFAULT_ATTRS static inline uint32_t __gpu_num_lanes() {
  return __builtin_amdgcn_wavefrontsize();
}

// Returns the id of the thread inside of an AMD wavefront executing together.
_DEFAULT_ATTRS [[clang::convergent]] static inline uint32_t __gpu_lane_id() {
  return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}

// Returns the bit-mask of active threads in the current wavefront.
_DEFAULT_ATTRS [[clang::convergent]] static inline uint64_t __gpu_lane_mask() {
  return __builtin_amdgcn_read_exec();
}

// Copies the value from the first active thread in the wavefront to the rest.
_DEFAULT_ATTRS [[clang::convergent]] static inline uint32_t
__gpu_broadcast(uint64_t __lane_mask, uint32_t __x) {
  return __builtin_amdgcn_readfirstlane(__x);
}

// Returns a bitmask of threads in the current lane for which \p x is true.
_DEFAULT_ATTRS [[clang::convergent]] static inline uint64_t
__gpu_ballot(uint64_t __lane_mask, bool __x) {
  // The lane_mask & gives the nvptx semantics when lane_mask is a subset of
  // the active threads
  return __lane_mask & __builtin_amdgcn_ballot_w64(__x);
}

// Waits for all the threads in the block to converge and issues a fence.
_DEFAULT_ATTRS [[clang::convergent]] static inline void __gpu_sync_threads() {
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}

// Wait for all threads in the wavefront to converge, this is a noop on AMDGPU.
_DEFAULT_ATTRS [[clang::convergent]] static inline void
__gpu_sync_lane(uint64_t __lane_mask) {
  __builtin_amdgcn_wave_barrier();
}

// Shuffles the the lanes inside the wavefront according to the given index.
_DEFAULT_ATTRS [[clang::convergent]] static inline uint32_t
__gpu_shuffle_idx(uint64_t __lane_mask, uint32_t __idx, uint32_t __x) {
  return __builtin_amdgcn_ds_bpermute(__idx << 2, __x);
}

// Terminates execution of the associated wavefront.
_DEFAULT_ATTRS [[noreturn]] static inline void __gpu_exit() {
  __builtin_amdgcn_endpgm();
}

#pragma omp end declare variant
#pragma omp end declare target

#endif // __AMDGPUINTRIN_H
