//===-- Implementation of qsort -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/qsort.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/stdlib/qsort_util.h"

#include <stdint.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, qsort,
                   (void *array, size_t array_size, size_t elem_size,
                    int (*compare)(const void *, const void *))) {
  if (array == nullptr || array_size == 0 || elem_size == 0)
    return;
  internal::Comparator c(compare);

  auto arr = internal::Array(reinterpret_cast<uint8_t *>(array), array_size,
                             elem_size, c);

  internal::sort(arr);
}

} // namespace LIBC_NAMESPACE_DECL
