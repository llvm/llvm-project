//===-- Unittests for heap sort -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SortingTest.h"
#include "src/stdlib/heap_sort.h"

void sort(const LIBC_NAMESPACE::internal::Array &array) {
  LIBC_NAMESPACE::internal::heap_sort(array);
}

LIST_SORTING_TESTS(HeapSort, sort);
