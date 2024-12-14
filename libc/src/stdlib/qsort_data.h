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

#include <stdint.h>

namespace LIBC_NAMESPACE_DECL {
namespace internal {
class Array {
  uint8_t *array_base;
  size_t array_len;
  size_t elem_size;

  uint8_t *get_internal(size_t i) const noexcept {
    return array_base + (i * elem_size);
  }

public:
  Array(uint8_t *a, size_t s, size_t e) noexcept
      : array_base(a), array_len(s), elem_size(e) {}

  inline void *get(size_t i) const noexcept {
    return reinterpret_cast<void *>(get_internal(i));
  }

  void swap(size_t i, size_t j) const noexcept {
    uint8_t *elem_i = get_internal(i);
    uint8_t *elem_j = get_internal(j);

    for (size_t b = 0; b < elem_size; ++b) {
      uint8_t temp = elem_i[b];
      elem_i[b] = elem_j[b];
      elem_j[b] = temp;
    }
  }

  size_t len() const noexcept { return array_len; }

  // Make an Array starting at index |i| and length |s|.
  inline Array make_array(size_t i, size_t s) const noexcept {
    return Array(get_internal(i), s, elem_size);
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
