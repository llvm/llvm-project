//===-- Implementation header for qsort utilities ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_QUICK_SORT_H
#define LLVM_LIBC_SRC_STDLIB_QUICK_SORT_H

#include "src/__support/CPP/cstddef.h"
#include "src/__support/macros/config.h"
#include "src/stdlib/qsort_pivot.h"

#include <stdint.h>

namespace LIBC_NAMESPACE_DECL {
namespace internal {

// A simple quicksort implementation using the Hoare partition scheme.
template <typename F>
size_t partition_hoare(const Array &array, const void *pivot,
                       const F &is_less) {
  const size_t array_len = array.len();

  size_t left = 0;
  size_t right = array_len;

  while (true) {
    while (left < right && is_less(array.get(left), pivot))
      ++left;

    while (true) {
      --right;
      if (left >= right || is_less(array.get(right), pivot)) {
        break;
      }
    }

    if (left >= right)
      break;

    array.swap(left, right);
    ++left;
  }

  return left;
}

template <typename F>
size_t partition(const Array &array, size_t pivot_index, const F &is_less) {
  // Place the pivot at the beginning of the array.
  array.swap(0, pivot_index);

  const Array array_without_pivot = array.make_array(1, array.len() - 1);
  const void *pivot = array.get(0);
  const size_t num_lt = partition_hoare(array_without_pivot, pivot, is_less);

  // Place the pivot between the two partitions.
  array.swap(0, num_lt);

  return num_lt;
}

template <typename F> void quick_sort(Array &array, const F &is_less) {
  while (true) {
    const size_t array_len = array.len();
    if (array_len <= 1)
      return;

    const size_t pivot_index = choose_pivot(array, is_less);
    size_t split_index = partition(array, pivot_index, is_less);

    if (array_len == 2)
      // The partition operation sorts the two element array.
      return;

    // Split the array into `left`, `pivot`, and `right`.
    Array left = array.make_array(0, split_index);
    const size_t right_start = split_index + 1;
    Array right = array.make_array(right_start, array.len() - right_start);

    // Recurse to sort the smaller of the two, and then loop round within this
    // function to sort the larger. This way, recursive call depth is bounded
    // by log2 of the total array size, because every recursive call is sorting
    // a list at most half the length of the one in its caller.
    if (left.len() < right.len()) {
      quick_sort(left, is_less);
      array.reset_bounds(right_start, right.len());
    } else {
      quick_sort(right, is_less);
      array.reset_bounds(0, left.len());
    }
  }
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDLIB_QUICK_SORT_H
