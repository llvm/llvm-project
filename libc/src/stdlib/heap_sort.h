//===-- Implementation of heap sort -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_HEAP_SORT_H
#define LLVM_LIBC_SRC_STDLIB_HEAP_SORT_H

#include "src/__support/CPP/cstddef.h"
#include "src/stdlib/qsort_data.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

// A simple in-place heapsort implementation.
// Follow the implementation in https://en.wikipedia.org/wiki/Heapsort.

template <typename A, typename F>
LIBC_INLINE void heap_sort(const A &array, const F &is_less) {
  size_t end = array.len();
  size_t start = end / 2;

  const auto left_child = [](size_t i) -> size_t { return 2 * i + 1; };

  while (end > 1) {
    if (start > 0) {
      // Select the next unheapified element to sift down.
      --start;
    } else {
      // Extract the max element of the heap, moving a leaf to root to be sifted
      // down.
      --end;
      array.swap(0, end);
    }

    // Sift start down the heap.
    size_t i, j;

    // Find the sift-down path all the way to the leaves.
    for (i = start; j = left_child(i), j + 1 < end;)
      i = is_less(array.get(j), array.get(j + 1)) ? (j + 1) : j;

    // Special case for the last leaf with no sibling.
    if (j < end)
      i = j;

    // Backtrack to the correct location.
    while (i != start && is_less(array.get(i), array.get(start)))
      i = (i - 1) / 2;

    // Shift the element into its correct place.
    j = i;
    while (i != start) {
      i = (i - 1) / 2;
      array.swap(i, j);
    }
  }
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDLIB_HEAP_SORT_H
