//===-- heap_sort_fuzz.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Fuzzing test for llvm-libc heap_sort implementation.
///
//===----------------------------------------------------------------------===//

#include "src/stdlib/qsort_util.h"
#include <stdint.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  const size_t array_size = size / sizeof(int);
  if (array_size == 0)
    return 0;

  int *array = new int[array_size];
  const int *data_as_int = reinterpret_cast<const int *>(data);
  for (size_t i = 0; i < array_size; ++i)
    array[i] = data_as_int[i];

  const auto is_less = [](const void *a_ptr,
                          const void *b_ptr) noexcept -> bool {
    const int &a = *static_cast<const int *>(a_ptr);
    const int &b = *static_cast<const int *>(b_ptr);

    return a < b;
  };

  constexpr bool USE_QUICKSORT = false;
  LIBC_NAMESPACE::internal::unstable_sort_impl<USE_QUICKSORT>(
      array, array_size, sizeof(int), is_less);

  for (size_t i = 0; i < array_size - 1; ++i) {
    if (array[i] > array[i + 1])
      __builtin_trap();
  }

  delete[] array;
  return 0;
}
