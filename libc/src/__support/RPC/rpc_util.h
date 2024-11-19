//===-- Shared memory RPC client / server utilities -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_RPC_RPC_UTIL_H
#define LLVM_LIBC_SRC___SUPPORT_RPC_RPC_UTIL_H

#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

#include <stddef.h>
#include <stdint.h>

#if defined(__NVPTX__) || defined(__AMDGPU__)
#include <gpuintrin.h>
#define RPC_TARGET_IS_GPU
#endif

namespace LIBC_NAMESPACE_DECL {
namespace rpc {

template <typename T> struct type_identity {
  using type = T;
};

template <class T, T v> struct type_constant {
  static inline constexpr T value = v;
};

template <class T> struct remove_reference : type_identity<T> {};
template <class T> struct remove_reference<T &> : type_identity<T> {};
template <class T> struct remove_reference<T &&> : type_identity<T> {};

template <class T> struct is_const : type_constant<bool, false> {};
template <class T> struct is_const<const T> : type_constant<bool, true> {};

/// Freestanding implementation of std::move.
template <class T>
LIBC_INLINE constexpr typename remove_reference<T>::type &&move(T &&t) {
  return static_cast<typename remove_reference<T>::type &&>(t);
}

/// Freestanding implementation of std::forward.
template <typename T>
LIBC_INLINE constexpr T &&forward(typename remove_reference<T>::type &value) {
  return static_cast<T &&>(value);
}
template <typename T>
LIBC_INLINE constexpr T &&forward(typename remove_reference<T>::type &&value) {
  return static_cast<T &&>(value);
}

struct in_place_t {
  LIBC_INLINE explicit in_place_t() = default;
};

struct nullopt_t {
  LIBC_INLINE constexpr explicit nullopt_t() = default;
};

constexpr inline in_place_t in_place{};
constexpr inline nullopt_t nullopt{};

/// Freestanding and minimal implementation of std::optional.
template <typename T> class optional {
  template <typename U> struct OptionalStorage {
    union {
      char empty;
      U stored_value;
    };

    bool in_use = false;

    LIBC_INLINE ~OptionalStorage() { reset(); }

    LIBC_INLINE constexpr OptionalStorage() : empty() {}

    template <typename... Args>
    LIBC_INLINE constexpr explicit OptionalStorage(in_place_t, Args &&...args)
        : stored_value(forward<Args>(args)...) {}

    LIBC_INLINE constexpr void reset() {
      if (in_use)
        stored_value.~U();
      in_use = false;
    }
  };

  OptionalStorage<T> storage;

public:
  LIBC_INLINE constexpr optional() = default;
  LIBC_INLINE constexpr optional(nullopt_t) {}

  LIBC_INLINE constexpr optional(const T &t) : storage(in_place, t) {
    storage.in_use = true;
  }
  LIBC_INLINE constexpr optional(const optional &) = default;

  LIBC_INLINE constexpr optional(T &&t) : storage(in_place, move(t)) {
    storage.in_use = true;
  }
  LIBC_INLINE constexpr optional(optional &&O) = default;

  LIBC_INLINE constexpr optional &operator=(T &&t) {
    storage = move(t);
    return *this;
  }
  LIBC_INLINE constexpr optional &operator=(optional &&) = default;

  LIBC_INLINE constexpr optional &operator=(const T &t) {
    storage = t;
    return *this;
  }
  LIBC_INLINE constexpr optional &operator=(const optional &) = default;

  LIBC_INLINE constexpr void reset() { storage.reset(); }

  LIBC_INLINE constexpr const T &value() const & {
    return storage.stored_value;
  }

  LIBC_INLINE constexpr T &value() & { return storage.stored_value; }

  LIBC_INLINE constexpr explicit operator bool() const {
    return storage.in_use;
  }
  LIBC_INLINE constexpr bool has_value() const { return storage.in_use; }
  LIBC_INLINE constexpr const T *operator->() const {
    return &storage.stored_value;
  }
  LIBC_INLINE constexpr T *operator->() { return &storage.stored_value; }
  LIBC_INLINE constexpr const T &operator*() const & {
    return storage.stored_value;
  }
  LIBC_INLINE constexpr T &operator*() & { return storage.stored_value; }

  LIBC_INLINE constexpr T &&value() && { return move(storage.stored_value); }
  LIBC_INLINE constexpr T &&operator*() && {
    return move(storage.stored_value);
  }
};

/// Suspend the thread briefly to assist the thread scheduler during busy loops.
LIBC_INLINE void sleep_briefly() {
#if defined(LIBC_TARGET_ARCH_IS_NVPTX)
  if (__nvvm_reflect("__CUDA_ARCH") >= 700)
    asm("nanosleep.u32 64;" ::: "memory");
#elif defined(LIBC_TARGET_ARCH_IS_AMDGPU)
  __builtin_amdgcn_s_sleep(2);
#elif defined(LIBC_TARGET_ARCH_IS_X86)
  __builtin_ia32_pause();
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64) && __has_builtin(__builtin_arm_isb)
  __builtin_arm_isb(0xf);
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
  asm volatile("isb\n" ::: "memory");
#else
  // Simply do nothing if sleeping isn't supported on this platform.
#endif
}

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
  if constexpr (is_const<T>::value)
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
