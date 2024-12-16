//===-- Unittests for heap sort -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SortingTest.h"
#include "src/stdlib/qsort_util.h"

void heap_sort(void *array, size_t array_size, size_t elem_size,
               int (*compare)(const void *, const void *)) {

  if (array == nullptr || array_size == 0 || elem_size == 0)
    return;

  auto arr = LIBC_NAMESPACE::internal::ArrayGenericSize(
      reinterpret_cast<uint8_t *>(array), array_size, elem_size);

  LIBC_NAMESPACE::internal::heap_sort(
      arr, [compare](const void *a, const void *b) noexcept -> bool {
        return compare(a, b) < 0;
      });
}

LIST_SORTING_TESTS(HeapSort, heap_sort);
