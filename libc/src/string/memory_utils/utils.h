//===-- Memory utils --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MEMORY_UTILS_UTILS_H
#define LLVM_LIBC_SRC_MEMORY_UTILS_UTILS_H

#include "src/__support/architectures.h"

// Cache line sizes for ARM: These values are not strictly correct since
// cache line sizes depend on implementations, not architectures.  There
// are even implementations with cache line sizes configurable at boot
// time.
#if defined(LLVM_LIBC_ARCH_AARCH64) || defined(LLVM_LIBC_ARCH_X86)
#define LLVM_LIBC_CACHELINE_SIZE 64
#elif defined(LLVM_LIBC_ARCH_ARM)
#define LLVM_LIBC_CACHELINE_SIZE 32
#else
#error "Unsupported platform for memory functions."
#endif

#include <stddef.h> // size_t
#include <stdint.h> // intptr_t / uintptr_t

namespace __llvm_libc {

// Allows compile time error reporting in `if constexpr` branches.
template <bool flag = false>
static void deferred_static_assert(const char *msg) {
  static_assert(flag, "compilation error");
  (void)msg;
}

// Return whether `value` is zero or a power of two.
static constexpr bool is_power2_or_zero(size_t value) {
  return (value & (value - 1U)) == 0;
}

// Return whether `value` is a power of two.
static constexpr bool is_power2(size_t value) {
  return value && is_power2_or_zero(value);
}

// Compile time version of log2 that handles 0.
static constexpr size_t log2(size_t value) {
  return (value == 0 || value == 1) ? 0 : 1 + log2(value / 2);
}

// Returns the first power of two preceding value or value if it is already a
// power of two (or 0 when value is 0).
static constexpr size_t le_power2(size_t value) {
  return value == 0 ? value : 1ULL << log2(value);
}

// Returns the first power of two following value or value if it is already a
// power of two (or 0 when value is 0).
static constexpr size_t ge_power2(size_t value) {
  return is_power2_or_zero(value) ? value : 1ULL << (log2(value) + 1);
}

template <size_t alignment> intptr_t offset_from_last_aligned(const void *ptr) {
  static_assert(is_power2(alignment), "alignment must be a power of 2");
  return reinterpret_cast<uintptr_t>(ptr) & (alignment - 1U);
}

template <size_t alignment> intptr_t offset_to_next_aligned(const void *ptr) {
  static_assert(is_power2(alignment), "alignment must be a power of 2");
  // The logic is not straightforward and involves unsigned modulo arithmetic
  // but the generated code is as fast as it can be.
  return -reinterpret_cast<uintptr_t>(ptr) & (alignment - 1U);
}

// Returns the offset from `ptr` to the next cache line.
static inline intptr_t offset_to_next_cache_line(const void *ptr) {
  return offset_to_next_aligned<LLVM_LIBC_CACHELINE_SIZE>(ptr);
}

template <size_t alignment, typename T> static T *assume_aligned(T *ptr) {
  return reinterpret_cast<T *>(__builtin_assume_aligned(ptr, alignment));
}
#if defined __has_builtin
#if __has_builtin(__builtin_memcpy_inline)
#define LLVM_LIBC_HAS_BUILTIN_MEMCPY_INLINE
#endif
#endif

// Performs a constant count copy.
template <size_t Size>
static inline void memcpy_inline(void *__restrict dst,
                                 const void *__restrict src) {
#ifdef LLVM_LIBC_HAS_BUILTIN_MEMCPY_INLINE
  __builtin_memcpy_inline(dst, src, Size);
#else
  for (size_t i = 0; i < Size; ++i)
    static_cast<char *>(dst)[i] = static_cast<const char *>(src)[i];
#endif
}

using Ptr = char *;        // Pointer to raw data.
using CPtr = const char *; // Const pointer to raw data.

// Loads bytes from memory (possibly unaligned) and materializes them as type.
template <typename T> static inline T load(CPtr ptr) {
  T Out;
  memcpy_inline<sizeof(T)>(&Out, ptr);
  return Out;
}

// Stores a value of type T in memory (possibly unaligned)
template <typename T> static inline void store(Ptr ptr, T value) {
  memcpy_inline<sizeof(T)>(ptr, &value);
}

// For an operation like memset that operates on a pointer and a count, advances
// the pointer by offset bytes and decrease count by the same amount.
static inline void adjust(ptrdiff_t offset, Ptr &ptr, size_t &count) {
  ptr += offset;
  count -= offset;
}

// For an operation like memcpy or memcmp that operates on two pointers and a
// count, advances the pointers by offset bytes and decrease count by the same
// amount.
template <typename T1, typename T2>
static inline void adjust(ptrdiff_t offset, T1 *__restrict &p1,
                          T2 *__restrict &p2, size_t &count) {
  p1 += offset;
  p2 += offset;
  count -= offset;
}

// For an operation like memset that operates on a pointer and a count, advances
// the pointer so it is aligned to SIZE bytes and decrease count by the same
// amount.
// We make sure the compiler knows about the adjusted pointer alignment.
template <size_t SIZE> void align(Ptr &ptr, size_t &count) {
  adjust(offset_to_next_aligned<SIZE>(ptr), ptr, count);
  ptr = assume_aligned<SIZE>(ptr);
}

// For an operation like memcpy or memcmp that operates on two pointers and a
// count, advances the pointers so one of them gets aligned to SIZE bytes and
// decrease count by the same amount.
// We make sure the compiler knows about the adjusted pointer alignment.
enum class Arg { _1, _2, Dst = _1, Src = _2, Lhs = _1, Rhs = _2 };
template <size_t SIZE, Arg AlignOn, typename T1, typename T2>
void align(T1 *__restrict &p1, T2 *__restrict &p2, size_t &count) {
  if constexpr (AlignOn == Arg::_1) {
    adjust(offset_to_next_aligned<SIZE>(p1), p1, p2, count);
    p1 = assume_aligned<SIZE>(p1);
  } else if constexpr (AlignOn == Arg::_2) {
    adjust(offset_to_next_aligned<SIZE>(p2), p1, p2, count);
    p2 = assume_aligned<SIZE>(p2);
  } else {
    deferred_static_assert("AlignOn must be either Arg::_1 or Arg::_2");
  }
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_MEMORY_UTILS_UTILS_H
