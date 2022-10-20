//===-- Memory utils --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MEMORY_UTILS_UTILS_H
#define LLVM_LIBC_SRC_MEMORY_UTILS_UTILS_H

#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/type_traits.h"

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

// Returns the number of bytes to substract from ptr to get to the previous
// multiple of alignment. If ptr is already aligned returns 0.
template <size_t alignment> uintptr_t distance_to_align_down(const void *ptr) {
  static_assert(is_power2(alignment), "alignment must be a power of 2");
  return reinterpret_cast<uintptr_t>(ptr) & (alignment - 1U);
}

// Returns the number of bytes to add to ptr to get to the next multiple of
// alignment. If ptr is already aligned returns 0.
template <size_t alignment> uintptr_t distance_to_align_up(const void *ptr) {
  static_assert(is_power2(alignment), "alignment must be a power of 2");
  // The logic is not straightforward and involves unsigned modulo arithmetic
  // but the generated code is as fast as it can be.
  return -reinterpret_cast<uintptr_t>(ptr) & (alignment - 1U);
}

// Returns the number of bytes to add to ptr to get to the next multiple of
// alignment. If ptr is already aligned returns alignment.
template <size_t alignment>
uintptr_t distance_to_next_aligned(const void *ptr) {
  return alignment - distance_to_align_down<alignment>(ptr);
}

// Returns the same pointer but notifies the compiler that it is aligned.
template <size_t alignment, typename T> static T *assume_aligned(T *ptr) {
  return reinterpret_cast<T *>(__builtin_assume_aligned(ptr, alignment));
}

#if defined __has_builtin
#if __has_builtin(__builtin_memcpy_inline)
#define LLVM_LIBC_HAS_BUILTIN_MEMCPY_INLINE
#endif
#endif

#if defined __has_builtin
#if __has_builtin(__builtin_memset_inline)
#define LLVM_LIBC_HAS_BUILTIN_MEMSET_INLINE
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

// Loads bytes from memory (possibly unaligned) and materializes them as
// type.
template <typename T> static inline T load(CPtr ptr) {
  T Out;
  memcpy_inline<sizeof(T)>(&Out, ptr);
  return Out;
}

// Stores a value of type T in memory (possibly unaligned).
template <typename T> static inline void store(Ptr ptr, T value) {
  memcpy_inline<sizeof(T)>(ptr, &value);
}

// Advances the pointers p1 and p2 by offset bytes and decrease count by the
// same amount.
template <typename T1, typename T2>
static inline void adjust(ptrdiff_t offset, T1 *__restrict &p1,
                          T2 *__restrict &p2, size_t &count) {
  p1 += offset;
  p2 += offset;
  count -= offset;
}

// Advances p1 and p2 so p1 gets aligned to the next SIZE bytes boundary
// and decrease count by the same amount.
// We make sure the compiler knows about the adjusted pointer alignment.
template <size_t SIZE, typename T1, typename T2>
void align_p1_to_next_boundary(T1 *__restrict &p1, T2 *__restrict &p2,
                               size_t &count) {
  adjust(distance_to_next_aligned<SIZE>(p1), p1, p2, count);
  p1 = assume_aligned<SIZE>(p1);
}

// Same as align_p1_to_next_boundary above but with a single pointer instead.
template <size_t SIZE, typename T1>
void align_to_next_boundary(T1 *&p1, size_t &count) {
  CPtr dummy;
  align_p1_to_next_boundary<SIZE>(p1, dummy, count);
}

// An enum class that discriminates between the first and second pointer.
enum class Arg { P1, P2, Dst = P1, Src = P2 };

// Same as align_p1_to_next_boundary but allows for aligning p2 instead of p1.
// Precondition: &p1 != &p2
template <size_t SIZE, Arg AlignOn, typename T1, typename T2>
void align_to_next_boundary(T1 *__restrict &p1, T2 *__restrict &p2,
                            size_t &count) {
  if constexpr (AlignOn == Arg::P1)
    align_p1_to_next_boundary<SIZE>(p1, p2, count);
  else if constexpr (AlignOn == Arg::P2)
    align_p1_to_next_boundary<SIZE>(p2, p1, count); // swapping p1 and p2.
  else
    deferred_static_assert("AlignOn must be either Arg::P1 or Arg::P2");
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_MEMORY_UTILS_UTILS_H
