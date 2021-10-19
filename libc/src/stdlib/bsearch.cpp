//===-- Implementation of bsearch -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/bsearch.h"
#include "src/__support/common.h"

#include <stdint.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(void *, bsearch,
                   (const void *key, const void *array, size_t array_size,
                    size_t elem_size,
                    int (*compare)(const void *, const void *))) {
  if (key == nullptr || array == nullptr || array_size == 0 || elem_size == 0)
    return nullptr;

  while (array_size > 0) {
    size_t mid = array_size / 2;
    const void *elem =
        reinterpret_cast<const uint8_t *>(array) + mid * elem_size;
    int compare_result = compare(key, elem);
    if (compare_result == 0)
      return const_cast<void *>(elem);

    if (compare_result < 0) {
      // This means that key is less than the element at |mid|.
      // So, in the next iteration, we only compare elements less
      // than mid.
      array_size = mid;
    } else {
      // |mid| is strictly less than |array_size|. So, the below
      // decrement in |array_size| will not lead to a wrap around.
      array_size -= (mid + 1);
      array = reinterpret_cast<const uint8_t *>(elem) + elem_size;
    }
  }

  return nullptr;
}

} // namespace __llvm_libc
