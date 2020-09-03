//===-- Memory utils --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MEMORY_UTILS_H
#define LLVM_LIBC_SRC_MEMORY_UTILS_H

#include "src/string/memory_utils/cacheline_size.h"

#include <stddef.h> // size_t
#include <stdint.h> // intptr_t / uintptr_t

namespace __llvm_libc {

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

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_MEMORY_UTILS_H
