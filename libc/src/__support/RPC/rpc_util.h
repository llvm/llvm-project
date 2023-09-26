//===-- Shared memory RPC client / server utilities -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_RPC_RPC_UTILS_H
#define LLVM_LIBC_SRC___SUPPORT_RPC_RPC_UTILS_H

#include "src/__support/CPP/type_traits.h"
#include "src/__support/GPU/utils.h"
#include "src/__support/macros/attributes.h" // LIBC_INLINE
#include "src/__support/macros/properties/architectures.h"

namespace LIBC_NAMESPACE {
namespace rpc {

/// Suspend the thread briefly to assist the thread scheduler during busy loops.
LIBC_INLINE void sleep_briefly() {
#if defined(LIBC_TARGET_ARCH_IS_NVPTX) && __CUDA_ARCH__ >= 700
  LIBC_INLINE_ASM("nanosleep.u32 64;" ::: "memory");
#elif defined(LIBC_TARGET_ARCH_IS_AMDGPU)
  __builtin_amdgcn_s_sleep(2);
#elif defined(LIBC_TARGET_ARCH_IS_X86)
  __builtin_ia32_pause();
#else
  // Simply do nothing if sleeping isn't supported on this platform.
#endif
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

} // namespace rpc
} // namespace LIBC_NAMESPACE

#endif
