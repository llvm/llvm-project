//===-- Implementation header for qsort utilities ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_QSORT_UTIL_H
#define LLVM_LIBC_SRC_STDLIB_QSORT_UTIL_H

#include "src/__support/macros/attributes.h"
#include <stdint.h>
#include <stdlib.h>

namespace __llvm_libc::internal {

// A simple quicksort implementation using the Hoare partition scheme.

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

static size_t partition(const Array &array) {
  const size_t array_size = array.size();
  size_t pivot_index = array_size / 2;
  uint8_t *pivot = array.get(pivot_index);
  size_t i = 0;
  size_t j = array_size - 1;

  while (true) {
    int compare_i, compare_j;

    while ((compare_i = array.elem_compare(i, pivot)) < 0)
      ++i;
    while ((compare_j = array.elem_compare(j, pivot)) > 0)
      --j;

    // At some point i will crossover j so we will definitely break out of
    // this while loop.
    if (i >= j)
      return j + 1;

    array.swap(i, j);

    // The pivot itself might have got swapped so we will update the pivot.
    if (i == pivot_index) {
      pivot = array.get(j);
      pivot_index = j;
    } else if (j == pivot_index) {
      pivot = array.get(i);
      pivot_index = i;
    }

    if (compare_i == 0 && compare_j == 0) {
      // If we do not move the pointers, we will end up with an
      // infinite loop as i and j will be stuck without advancing.
      ++i;
      --j;
    }
  }
}

LIBC_INLINE void quicksort(const Array &array) {
  const size_t array_size = array.size();
  if (array_size <= 1)
    return;
  size_t split_index = partition(array);
  if (array_size <= 2) {
    // The partition operation sorts the two element array.
    return;
  }
  quicksort(array.make_array(0, split_index));
  quicksort(array.make_array(split_index, array.size() - split_index));
}

} // namespace __llvm_libc::internal

#endif // LLVM_LIBC_SRC_STDLIB_QSORT_UTIL_H
