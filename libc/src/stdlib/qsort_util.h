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

template <typename A, typename F> void sort_inst(A &array, const F &is_less) {
#if LIBC_QSORT_IMPL == LIBC_QSORT_QUICK_SORT
  quick_sort(array, is_less);
#elif LIBC_QSORT_IMPL == LIBC_QSORT_HEAP_SORT
  heap_sort(array, is_less);
#endif
}

template <typename F>
void unstable_sort(void *array, size_t array_len, size_t elem_size,
                   const F &is_less) {
  if (array == nullptr || array_len == 0 || elem_size == 0) {
    return;
  }

  uint8_t *array_base = reinterpret_cast<uint8_t *>(array);

  switch (elem_size) {
  case 4: {
    auto arr_fixed_size = internal::ArrayFixedSize<4>(array_base, array_len);
    sort_inst(arr_fixed_size, is_less);
    return;
  }
  case 8: {
    auto arr_fixed_size = internal::ArrayFixedSize<8>(array_base, array_len);
    sort_inst(arr_fixed_size, is_less);
    return;
  }
  case 16: {
    auto arr_fixed_size = internal::ArrayFixedSize<16>(array_base, array_len);
    sort_inst(arr_fixed_size, is_less);
    return;
  }
  default:
    auto arr_generic_size =
        internal::ArrayGenericSize(array_base, array_len, elem_size);
    sort_inst(arr_generic_size, is_less);
    return;
  }
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDLIB_QSORT_UTIL_H
