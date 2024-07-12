//===-- Implementation of heap sort -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_HEAP_SORT_H
#define LLVM_LIBC_SRC_STDLIB_HEAP_SORT_H

#include "qsort_util.h"
#include "src/__support/CPP/cstddef.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

// A simple in-place heapsort implementation.
// Follow the implementation in https://en.wikipedia.org/wiki/Heapsort.

LIBC_INLINE void heap_sort(const Array &array) {
  size_t end = array.size();
  size_t start = end / 2;

  auto left_child = [](size_t i) -> size_t { return 2 * i + 1; };

  while (end > 1) {
    if (start > 0) {
      --start;
    } else {
      --end;
      array.swap(0, end);
    }

    size_t root = start;
    while (left_child(root) < end) {
      size_t child = left_child(root);
      if (child + 1 < end &&
          array.elem_compare(child, array.get(child + 1)) < 0)
        ++child;

      if (array.elem_compare(root, array.get(child)) >= 0)
        break;

      array.swap(root, child);
      root = child;
    }
  }
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDLIB_HEAP_SORT_H
