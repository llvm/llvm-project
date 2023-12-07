//===-- Implementation of qsort_r -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/qsort_r.h"
#include "src/__support/common.h"
#include "src/stdlib/qsort_util.h"

#include <stdint.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(void, qsort_r,
                   (void *array, size_t array_size, size_t elem_size,
                    int (*compare)(const void *, const void *, void *),
                    void *arg)) {
  if (array == nullptr || array_size == 0 || elem_size == 0)
    return;
  internal::Comparator c(compare, arg);
  internal::quicksort(internal::Array(reinterpret_cast<uint8_t *>(array),
                                      array_size, elem_size, c));
}

} // namespace __llvm_libc
