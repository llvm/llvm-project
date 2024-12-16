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
#include "src/__support/big_int.h"
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

template <typename F>
void quick_sort_impl(Array &array, const void *ancestor_pivot, size_t limit,
                     const F &is_less) {
  while (true) {
    const size_t array_len = array.len();
    if (array_len <= 1)
      return;

    // If too many bad pivot choices were made, simply fall back to
    // heapsort in order to guarantee `O(N x log(N))` worst-case.
    if (limit == 0) {
      heap_sort(array, is_less);
      return;
    }

    limit -= 1;

    const size_t pivot_index = choose_pivot(array, is_less);

    // If the chosen pivot is equal to the predecessor, then it's the smallest
    // element in the slice. Partition the slice into elements equal to and
    // elements greater than the pivot. This case is usually hit when the slice
    // contains many duplicate elements.
    if (ancestor_pivot) {
      if (!is_less(ancestor_pivot, array.get(pivot_index))) {
        const size_t num_lt =
            partition(array, pivot_index,
                      [is_less](const void *a, const void *b) noexcept -> bool {
                        return !is_less(b, a);
                      });

        // Continue sorting elements greater than the pivot. We know that
        // `num_lt` cont
        array.reset_bounds(num_lt + 1, array.len() - (num_lt + 1));
        ancestor_pivot = nullptr;
        continue;
      }
    }

    size_t split_index = partition(array, pivot_index, is_less);

    if (array_len == 2)
      // The partition operation sorts the two element array.
      return;

    // Split the array into `left`, `pivot`, and `right`.
    Array left = array.make_array(0, split_index);
    const void *pivot = array.get(split_index);
    const size_t right_start = split_index + 1;
    Array right = array.make_array(right_start, array.len() - right_start);

    // Recurse into the left side. We have a fixed recursion limit,
    // testing shows no real benefit for recursing into the shorter
    // side.
    quick_sort_impl(left, ancestor_pivot, limit, is_less);

    // Continue with the right side.
    array = right;
    ancestor_pivot = pivot;
  }
}

constexpr size_t ilog2(size_t n) { return cpp::bit_width(n) - 1; }

template <typename F> void quick_sort(Array &array, const F &is_less) {
  const void *ancestor_pivot = nullptr;
  // Limit the number of imbalanced partitions to `2 * floor(log2(len))`.
  // The binary OR by one is used to eliminate the zero-check in the logarithm.
  const size_t limit = 2 * ilog2((array.len() | 1));
  quick_sort_impl(array, ancestor_pivot, limit, is_less);
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDLIB_QUICK_SORT_H
