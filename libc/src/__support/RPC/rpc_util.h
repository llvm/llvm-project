//===-- Shared memory RPC client / server utilities -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_RPC_RPC_UTILS_H
#define LLVM_LIBC_SRC_SUPPORT_RPC_RPC_UTILS_H

#include "src/__support/GPU/utils.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/properties/architectures.h"

namespace __llvm_libc {
namespace rpc {

/// Maximum amount of data a single lane can use.
constexpr uint64_t MAX_LANE_SIZE = 64;

/// Suspend the thread briefly to assist the thread scheduler during busy loops.
LIBC_INLINE void sleep_briefly() {
#if defined(LIBC_TARGET_ARCH_IS_NVPTX) && __CUDA_ARCH__ >= 700
  asm("nanosleep.u32 64;" ::: "memory");
#elif defined(LIBC_TARGET_ARCH_IS_AMDGPU)
  __builtin_amdgcn_s_sleep(2);
#else
  // Simply do nothing if sleeping isn't supported on this platform.
#endif
}

/// Get the first active thread inside the lane.
LIBC_INLINE uint64_t get_first_lane_id(uint64_t lane_mask) {
  return __builtin_ffsl(lane_mask) - 1;
}

/// Conditional that is only true for a single thread in a lane.
LIBC_INLINE bool is_first_lane(uint64_t lane_mask) {
  return gpu::get_lane_id() == get_first_lane_id(lane_mask);
}

/// Conditional to indicate if this process is running on the GPU.
LIBC_INLINE constexpr bool is_process_gpu() {
#if defined(LIBC_TARGET_ARCH_IS_GPU)
  return true;
#else
  return false;
#endif
}

/// Return \p val aligned "upwards" according to \p align.
template <typename V, typename A> LIBC_INLINE V align_up(V val, A align) {
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

/// Helper to get the maximum value.
template <typename T> LIBC_INLINE const T &max(const T &x, const T &y) {
  return x < y ? y : x;
}

} // namespace rpc
} // namespace __llvm_libc

#endif
