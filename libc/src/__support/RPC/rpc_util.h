//===-- Shared memory RPC client / server utilities -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_RPC_RPC_UTIL_H
#define LLVM_LIBC_SRC___SUPPORT_RPC_RPC_UTIL_H

#include "src/__support/CPP/type_traits.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/sleep.h"

#if defined(__NVPTX__) || defined(__AMDGPU__)
#include <gpuintrin.h>
#define RPC_TARGET_IS_GPU
#endif

namespace LIBC_NAMESPACE_DECL {
namespace rpc {

/// Conditional to indicate if this process is running on the GPU.
LIBC_INLINE constexpr bool is_process_gpu() {
#ifdef RPC_TARGET_IS_GPU
  return true;
#else
  return false;
#endif
}

/// Wait for all lanes in the group to complete.
LIBC_INLINE void sync_lane(uint64_t lane_mask) {
#ifdef RPC_TARGET_IS_GPU
  return __gpu_sync_lane(lane_mask);
#endif
}

/// Copies the value from the first active thread to the rest.
LIBC_INLINE uint32_t broadcast_value(uint64_t lane_mask, uint32_t x) {
#ifdef RPC_TARGET_IS_GPU
  return __gpu_read_first_lane_u32(lane_mask, x);
#else
  return x;
#endif
}

/// Returns the number lanes that participate in the RPC interface.
LIBC_INLINE uint32_t get_num_lanes() {
#ifdef RPC_TARGET_IS_GPU
  return __gpu_num_lanes();
#else
  return 1;
#endif
}

/// Returns the id of the thread inside of an AMD wavefront executing together.
LIBC_INLINE uint64_t get_lane_mask() {
#ifdef RPC_TARGET_IS_GPU
  return __gpu_lane_mask();
#else
  return 1;
#endif
}

/// Returns the id of the thread inside of an AMD wavefront executing together.
LIBC_INLINE uint32_t get_lane_id() {
#ifdef RPC_TARGET_IS_GPU
  return __gpu_lane_id();
#else
  return 0;
#endif
}

/// Conditional that is only true for a single thread in a lane.
LIBC_INLINE bool is_first_lane(uint64_t lane_mask) {
#ifdef RPC_TARGET_IS_GPU
  return __gpu_is_first_in_lane(lane_mask);
#else
  return true;
#endif
}

/// Returns a bitmask of threads in the current lane for which \p x is true.
LIBC_INLINE uint64_t ballot(uint64_t lane_mask, bool x) {
#ifdef RPC_TARGET_IS_GPU
  return __gpu_ballot(lane_mask, x);
#else
  return x;
#endif
}

/// Return \p val aligned "upwards" according to \p align.
template <typename V, typename A>
LIBC_INLINE constexpr V align_up(V val, A align) {
  return ((val + V(align) - 1) / V(align)) * V(align);
}

/// Utility to provide a unified interface between the CPU and GPU's memory
/// model. On the GPU stack variables are always private to a lane so we can
/// simply use the variable passed in. On the CPU we need to allocate enough
/// space for the whole lane and index into it.
template <typename V> LIBC_INLINE V &lane_value(V *val, uint32_t id) {
  if constexpr (is_process_gpu())
    return *val;
  return val[id];
}

/// Advance the \p p by \p bytes.
template <typename T, typename U> LIBC_INLINE T *advance(T *ptr, U bytes) {
  if constexpr (cpp::is_const_v<T>)
    return reinterpret_cast<T *>(reinterpret_cast<const uint8_t *>(ptr) +
                                 bytes);
  else
    return reinterpret_cast<T *>(reinterpret_cast<uint8_t *>(ptr) + bytes);
}

/// Wrapper around the optimal memory copy implementation for the target.
LIBC_INLINE void rpc_memcpy(void *dst, const void *src, size_t count) {
  __builtin_memcpy(dst, src, count);
}

template <class T> LIBC_INLINE constexpr const T &max(const T &a, const T &b) {
  return (a < b) ? b : a;
}

} // namespace rpc
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_RPC_RPC_UTIL_H
