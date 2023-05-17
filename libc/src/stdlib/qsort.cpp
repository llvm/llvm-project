//===-- Implementation of qsort -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/qsort.h"
#include "src/__support/common.h"

#include <stdint.h>

namespace __llvm_libc {

namespace internal {

// A simple quicksort implementation using the Hoare partition scheme.

class Array {
  typedef int (*comparator)(const void *, const void *);

  uint8_t *array;
  size_t array_size;
  size_t elem_size;
  comparator compare;

public:
  Array(uint8_t *a, size_t s, size_t e, comparator c)
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
    return compare(get(i), other);
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

static void quicksort(const Array &array) {
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

} // namespace internal

LLVM_LIBC_FUNCTION(void, qsort,
                   (void *array, size_t array_size, size_t elem_size,
                    int (*compare)(const void *, const void *))) {
  if (array == nullptr || array_size == 0 || elem_size == 0)
    return;
  internal::quicksort(internal::Array(reinterpret_cast<uint8_t *>(array),
                                      array_size, elem_size, compare));
}

} // namespace __llvm_libc
