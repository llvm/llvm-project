//===-- Data structures for sorting routines --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_QSORT_DATA_H
#define LLVM_LIBC_SRC_STDLIB_QSORT_DATA_H

#include "src/__support/CPP/cstddef.h"
#include "src/__support/macros/config.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "src/string/memory_utils/inline_memmove.h"

#include <stdint.h>

namespace LIBC_NAMESPACE_DECL {
namespace internal {

// Returns the max amount of bytes deemed reasonable - based on the target
// properties - for use in local stack arrays.
constexpr size_t max_stack_array_size() {
  // Uses target pointer size as heuristic how much memory is available and
  // unlikely to run into stack overflow and perf problems.
  constexpr size_t ptr_diff_size = sizeof(ptrdiff_t);

  if constexpr (ptr_diff_size >= 8) {
    return 4096;
  }

  if constexpr (ptr_diff_size == 4) {
    return 512;
  }

  // 8-bit platforms are just not gonna work well with libc, qsort
  // won't be the problem.
  // 16-bit platforms ought to be able to store 64 bytes on the stack.
  return 64;
}

class ArrayGenericSize {
  uint8_t *array_base;
  size_t array_len;
  size_t elem_size;

  uint8_t *get_internal(size_t i) const noexcept {
    return array_base + (i * elem_size);
  }

public:
  ArrayGenericSize(uint8_t *a, size_t s, size_t e) noexcept
      : array_base(a), array_len(s), elem_size(e) {}

  static constexpr bool has_fixed_size() { return false; }

  inline void *get(size_t i) const noexcept {
    return reinterpret_cast<void *>(get_internal(i));
  }

  void swap(size_t i, size_t j) const noexcept {
    // For sizes below this doing the extra function call is not
    // worth it.
    constexpr size_t MIN_MEMCPY_SIZE = 32;

    constexpr size_t STACK_ARRAY_SIZE = max_stack_array_size();
    alignas(32) uint8_t tmp[STACK_ARRAY_SIZE];

    uint8_t *elem_i = get_internal(i);
    uint8_t *elem_j = get_internal(j);

    if (elem_size >= MIN_MEMCPY_SIZE && elem_size <= STACK_ARRAY_SIZE) {
      // Block copies are much more efficient, even if `elem_size`
      // is unknown once `elem_size` passes a certain CPU specific
      // threshold.
      inline_memcpy(tmp, elem_i, elem_size);
      inline_memmove(elem_i, elem_j, elem_size);
      inline_memcpy(elem_j, tmp, elem_size);
    } else {
      for (size_t b = 0; b < elem_size; ++b) {
        uint8_t temp = elem_i[b];
        elem_i[b] = elem_j[b];
        elem_j[b] = temp;
      }
    }
  }

  size_t len() const noexcept { return array_len; }

  // Make an Array starting at index |i| and length |s|.
  inline ArrayGenericSize make_array(size_t i, size_t s) const noexcept {
    return ArrayGenericSize(get_internal(i), s, elem_size);
  }

  // Reset this Array to point at a different interval of the same
  // items starting at index |i|.
  inline void reset_bounds(size_t i, size_t s) noexcept {
    array_base = get_internal(i);
    array_len = s;
  }
};

// Having a specialized Array type for sorting that knowns at
// compile-time what the size of the element is, allows for much more
// efficient swapping and for cheaper offset calculations.
template <size_t ELEM_SIZE> class ArrayFixedSize {
  uint8_t *array_base;
  size_t array_len;

  uint8_t *get_internal(size_t i) const noexcept {
    return array_base + (i * ELEM_SIZE);
  }

public:
  ArrayFixedSize(uint8_t *a, size_t s) noexcept : array_base(a), array_len(s) {}

  // Beware this function is used a heuristic for cheap to swap types, so
  // instantiating `ArrayFixedSize` with `ELEM_SIZE > 100` is probably a bad
  // idea perf wise.
  static constexpr bool has_fixed_size() { return true; }

  inline void *get(size_t i) const noexcept {
    return reinterpret_cast<void *>(get_internal(i));
  }

  void swap(size_t i, size_t j) const noexcept {
    alignas(32) uint8_t tmp[ELEM_SIZE];

    uint8_t *elem_i = get_internal(i);
    uint8_t *elem_j = get_internal(j);

    inline_memcpy(tmp, elem_i, ELEM_SIZE);
    inline_memmove(elem_i, elem_j, ELEM_SIZE);
    inline_memcpy(elem_j, tmp, ELEM_SIZE);
  }

  size_t len() const noexcept { return array_len; }

  // Make an Array starting at index |i| and length |s|.
  inline ArrayFixedSize<ELEM_SIZE> make_array(size_t i,
                                              size_t s) const noexcept {
    return ArrayFixedSize<ELEM_SIZE>(get_internal(i), s);
  }

  // Reset this Array to point at a different interval of the same
  // items starting at index |i|.
  inline void reset_bounds(size_t i, size_t s) noexcept {
    array_base = get_internal(i);
    array_len = s;
  }
};

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDLIB_QSORT_DATA_H
