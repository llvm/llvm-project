//===-- Unittests for qsort -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SortingTest.h"
#include "src/stdlib/qsort.h"

void sort(const LIBC_NAMESPACE::internal::Array &array) {
  LIBC_NAMESPACE::qsort(reinterpret_cast<void *>(array.get(0)), array.size(),
                        sizeof(int), SortingTest::int_compare);
}

LIST_SORTING_TESTS(Qsort, sort);
