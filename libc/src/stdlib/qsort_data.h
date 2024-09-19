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

using Compare = int(const void *, const void *);
using CompareWithState = int(const void *, const void *, void *);

enum class CompType { COMPARE, COMPARE_WITH_STATE };

struct Comparator {
  union {
    Compare *comp_func;
    CompareWithState *comp_func_r;
  };
  const CompType comp_type;

  void *arg;

  Comparator(Compare *func)
      : comp_func(func), comp_type(CompType::COMPARE), arg(nullptr) {}

  Comparator(CompareWithState *func, void *arg_val)
      : comp_func_r(func), comp_type(CompType::COMPARE_WITH_STATE),
        arg(arg_val) {}

#if defined(__clang__)
  // Recent upstream changes to -fsanitize=function find more instances of
  // function type mismatches. One case is with the comparator passed to this
  // class. Libraries will tend to pass comparators that take pointers to
  // varying types while this comparator expects to accept const void pointers.
  // Ideally those tools would pass a function that strictly accepts const
  // void*s to avoid UB, or would use qsort_r to pass their own comparator.
  [[clang::no_sanitize("function")]]
#endif
  int comp_vals(const void *a, const void *b) const {
    if (comp_type == CompType::COMPARE) {
      return comp_func(a, b);
    } else {
      return comp_func_r(a, b, arg);
    }
  }
};

class Array {
  uint8_t *array;
  size_t array_size;
  size_t elem_size;
  Comparator compare;

public:
  Array(uint8_t *a, size_t s, size_t e, Comparator c)
      : array(a), array_size(s), elem_size(e), compare(c) {}

  uint8_t *get(size_t i) const { return array + i * elem_size; }

  void swap(size_t i, size_t j) const {
    uint8_t *elem_i = get(i);
    uint8_t *elem_j = get(j);
    for (size_t b = 0; b < elem_size; ++b) {
      uint8_t temp = elem_i[b];
      elem_i[b] = elem_j[b];
      elem_j[b] = temp;
    }
  }

  int elem_compare(size_t i, const uint8_t *other) const {
    // An element must compare equal to itself so we don't need to consult the
    // user provided comparator.
    if (get(i) == other)
      return 0;
    return compare.comp_vals(get(i), other);
  }

  size_t size() const { return array_size; }

  // Make an Array starting at index |i| and size |s|.
  Array make_array(size_t i, size_t s) const {
    return Array(get(i), s, elem_size, compare);
  }
};

using SortingRoutine = void(const Array &);

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDLIB_QSORT_DATA_H
