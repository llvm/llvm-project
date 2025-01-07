//===-------------- AMDGPU implementation of GPU utils ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_GPU_AMDGPU_IO_H
#define LLVM_LIBC_SRC___SUPPORT_GPU_AMDGPU_IO_H

#include "src/__support/common.h"
#include "src/__support/macros/config.h"

#include <stdint.h>

namespace LIBC_NAMESPACE_DECL {
namespace gpu {

/// Type aliases to the address spaces used by the AMDGPU backend.
template <typename T> using Private = [[clang::opencl_private]] T;
template <typename T> using Constant = [[clang::opencl_constant]] T;
template <typename T> using Local = [[clang::opencl_local]] T;
template <typename T> using Global = [[clang::opencl_global]] T;

/// Returns the number of workgroups in the 'x' dimension of the grid.
LIBC_INLINE uint32_t get_num_blocks_x() {
  return __builtin_amdgcn_grid_size_x() / __builtin_amdgcn_workgroup_size_x();
}

/// Returns the number of workgroups in the 'y' dimension of the grid.
LIBC_INLINE uint32_t get_num_blocks_y() {
  return __builtin_amdgcn_grid_size_y() / __builtin_amdgcn_workgroup_size_y();
}

/// Returns the number of workgroups in the 'z' dimension of the grid.
LIBC_INLINE uint32_t get_num_blocks_z() {
  return __builtin_amdgcn_grid_size_z() / __builtin_amdgcn_workgroup_size_z();
}

/// Returns the total number of workgruops in the grid.
LIBC_INLINE uint64_t get_num_blocks() {
  return get_num_blocks_x() * get_num_blocks_y() * get_num_blocks_z();
}

/// Returns the 'x' dimension of the current AMD workgroup's id.
LIBC_INLINE uint32_t get_block_id_x() {
  return __builtin_amdgcn_workgroup_id_x();
}

/// Returns the 'y' dimension of the current AMD workgroup's id.
LIBC_INLINE uint32_t get_block_id_y() {
  return __builtin_amdgcn_workgroup_id_y();
}

/// Returns the 'z' dimension of the current AMD workgroup's id.
LIBC_INLINE uint32_t get_block_id_z() {
  return __builtin_amdgcn_workgroup_id_z();
}

/// Returns the absolute id of the AMD workgroup.
LIBC_INLINE uint64_t get_block_id() {
  return get_block_id_x() + get_num_blocks_x() * get_block_id_y() +
         get_num_blocks_x() * get_num_blocks_y() * get_block_id_z();
}

/// Returns the number of workitems in the 'x' dimension.
LIBC_INLINE uint32_t get_num_threads_x() {
  return __builtin_amdgcn_workgroup_size_x();
}

/// Returns the number of workitems in the 'y' dimension.
LIBC_INLINE uint32_t get_num_threads_y() {
  return __builtin_amdgcn_workgroup_size_y();
}

/// Returns the number of workitems in the 'z' dimension.
LIBC_INLINE uint32_t get_num_threads_z() {
  return __builtin_amdgcn_workgroup_size_z();
}

/// Returns the total number of workitems in the workgroup.
LIBC_INLINE uint64_t get_num_threads() {
  return get_num_threads_x() * get_num_threads_y() * get_num_threads_z();
}

/// Returns the 'x' dimension id of the workitem in the current AMD workgroup.
LIBC_INLINE uint32_t get_thread_id_x() {
  return __builtin_amdgcn_workitem_id_x();
}

/// Returns the 'y' dimension id of the workitem in the current AMD workgroup.
LIBC_INLINE uint32_t get_thread_id_y() {
  return __builtin_amdgcn_workitem_id_y();
}

/// Returns the 'z' dimension id of the workitem in the current AMD workgroup.
LIBC_INLINE uint32_t get_thread_id_z() {
  return __builtin_amdgcn_workitem_id_z();
}

/// Returns the absolute id of the thread in the current AMD workgroup.
LIBC_INLINE uint64_t get_thread_id() {
  return get_thread_id_x() + get_num_threads_x() * get_thread_id_y() +
         get_num_threads_x() * get_num_threads_y() * get_thread_id_z();
}

/// Returns the size of an AMD wavefront, either 32 or 64 depending on hardware
/// and compilation options.
LIBC_INLINE uint32_t get_lane_size() {
  return __builtin_amdgcn_wavefrontsize();
}

/// Returns the id of the thread inside of an AMD wavefront executing together.
[[clang::convergent]] LIBC_INLINE uint32_t get_lane_id() {
  return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}

/// Returns the bit-mask of active threads in the current wavefront.
[[clang::convergent]] LIBC_INLINE uint64_t get_lane_mask() {
  return __builtin_amdgcn_read_exec();
}

/// Copies the value from the first active thread in the wavefront to the rest.
[[clang::convergent]] LIBC_INLINE uint32_t broadcast_value(uint64_t,
                                                           uint32_t x) {
  return __builtin_amdgcn_readfirstlane(x);
}

/// Returns a bitmask of threads in the current lane for which \p x is true.
[[clang::convergent]] LIBC_INLINE uint64_t ballot(uint64_t lane_mask, bool x) {
  // the lane_mask & gives the nvptx semantics when lane_mask is a subset of
  // the active threads
  return lane_mask & __builtin_amdgcn_ballot_w64(x);
}

/// Waits for all the threads in the block to converge and issues a fence.
[[clang::convergent]] LIBC_INLINE void sync_threads() {
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}

/// Waits for all pending memory operations to complete in program order.
[[clang::convergent]] LIBC_INLINE void memory_fence() {
  __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "");
}

/// Wait for all threads in the wavefront to converge, this is a noop on AMDGPU.
[[clang::convergent]] LIBC_INLINE void sync_lane(uint64_t) {
  __builtin_amdgcn_wave_barrier();
}

/// Shuffles the the lanes inside the wavefront according to the given index.
[[clang::convergent]] LIBC_INLINE uint32_t shuffle(uint64_t, uint32_t idx,
                                                   uint32_t x) {
  return __builtin_amdgcn_ds_bpermute(idx << 2, x);
}

/// Returns the current value of the GPU's processor clock.
/// NOTE: The RDNA3 and RDNA2 architectures use a 20-bit cycle counter.
LIBC_INLINE uint64_t processor_clock() { return __builtin_readcyclecounter(); }

/// Returns a fixed-frequency timestamp. The actual frequency is dependent on
/// the card and can only be queried via the driver.
LIBC_INLINE uint64_t fixed_frequency_clock() {
  return __builtin_readsteadycounter();
}

/// Terminates execution of the associated wavefront.
[[noreturn]] LIBC_INLINE void end_program() { __builtin_amdgcn_endpgm(); }

/// Returns a unique identifier for the process cluster the current wavefront is
/// executing on. Here we use the identifier for the compute unit (CU) and
/// shader engine.
/// FIXME: Currently unimplemented on AMDGPU until we have a simpler interface
/// than the one at
/// https://github.com/ROCm/clr/blob/develop/hipamd/include/hip/amd_detail/amd_device_functions.h#L899
LIBC_INLINE uint32_t get_cluster_id() { return 0; }

} // namespace gpu
} // namespace LIBC_NAMESPACE_DECL

#endif
