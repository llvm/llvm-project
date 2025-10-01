//===-- Unittests for qsort -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SortingTest.h"
#include "src/stdlib/qsort_util.h"

namespace {

void quick_sort(void *array, size_t array_size, size_t elem_size,
                int (*compare)(const void *, const void *)) {
  constexpr bool USE_QUICKSORT = true;

  const auto is_less = [compare](const void *a,
                                 const void *b) noexcept -> bool {
    return compare(a, b) < 0;
  };

  LIBC_NAMESPACE::internal::unstable_sort_impl<USE_QUICKSORT>(
      array, array_size, elem_size, is_less);
}

LIST_SORTING_TESTS(Qsort, quick_sort);

} // namespace
