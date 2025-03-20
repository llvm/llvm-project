//===-- Implementation of qsort_r -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/qsort_r.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/stdlib/qsort_util.h"

#include <stdint.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, qsort_r,
                   (void *array, size_t array_size, size_t elem_size,
                    int (*compare)(const void *, const void *, void *),
                    void *arg)) {

  const auto is_less = [compare, arg](const void *a, const void *b) -> bool {
    return compare(a, b, arg) < 0;
  };

  internal::unstable_sort(array, array_size, elem_size, is_less);
}

} // namespace LIBC_NAMESPACE_DECL
