//===-- Implementation header for qsort utilities ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_QUICK_SORT_H
#define LLVM_LIBC_SRC_STDLIB_QUICK_SORT_H

#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/stdlib/qsort_data.h"

#include <stdint.h>

namespace LIBC_NAMESPACE_DECL {
namespace internal {

// A simple quicksort implementation using the Hoare partition scheme.
LIBC_INLINE size_t partition(const Array &array) {
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

LIBC_INLINE void quick_sort(Array array) {
  while (true) {
    const size_t array_size = array.size();
    if (array_size <= 1)
      return;
    size_t split_index = partition(array);
    if (array_size == 2)
      // The partition operation sorts the two element array.
      return;

    // Make Arrays describing the two sublists that still need sorting.
    Array left = array.make_array(0, split_index);
    Array right = array.make_array(split_index, array.size() - split_index);

    // Recurse to sort the smaller of the two, and then loop round within this
    // function to sort the larger. This way, recursive call depth is bounded
    // by log2 of the total array size, because every recursive call is sorting
    // a list at most half the length of the one in its caller.
    if (left.size() < right.size()) {
      quick_sort(left);
      array.reset_bounds(right.get(0), right.size());
    } else {
      quick_sort(right);
      array.reset_bounds(left.get(0), left.size());
    }
  }
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDLIB_QUICK_SORT_H
