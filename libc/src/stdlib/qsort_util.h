//===-- Implementation header for qsort utilities ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_QSORT_UTIL_H
#define LLVM_LIBC_SRC_STDLIB_QSORT_UTIL_H

#include "src/stdlib/heap_sort.h"
#include "src/stdlib/quick_sort.h"

#define LIBC_QSORT_QUICK_SORT 1
#define LIBC_QSORT_HEAP_SORT 2

#ifndef LIBC_QSORT_IMPL
#define LIBC_QSORT_IMPL LIBC_QSORT_QUICK_SORT
#endif // LIBC_QSORT_IMPL

#if (LIBC_QSORT_IMPL != LIBC_QSORT_QUICK_SORT &&                               \
     LIBC_QSORT_IMPL != LIBC_QSORT_HEAP_SORT)
#error "LIBC_QSORT_IMPL is not recognized."
#endif

namespace LIBC_NAMESPACE_DECL {
namespace internal {

#if LIBC_QSORT_IMPL == LIBC_QSORT_QUICK_SORT
constexpr auto sort = quick_sort;
#elif LIBC_QSORT_IMPL == LIBC_QSORT_HEAP_SORT
constexpr auto sort = heap_sort;
#endif

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDLIB_QSORT_UTIL_H
