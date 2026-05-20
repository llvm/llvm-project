//===-- Implementation of qsort -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/qsort_s.h"
#define __STDC_WANT_LIB_EXT1__ 1
#include "hdr/stdint_proxy.h"
#undef __STDC_WANT_LIB_EXT1__
#include "src/__support/common.h"
#include "src/__support/constraint_handler.h"
#include "src/__support/macros/config.h"
#include "src/stdlib/qsort_util.h"

#define ERRNO_T_FAIL 1

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(errno_t, qsort_s,
                   (void *array, rsize_t array_size, rsize_t elem_size,
                    int (*compare)(const void *, const void *, void *),
                    void *context)) {
  const char *constraint_violation_msg = 0;
  if (array_size > RSIZE_MAX) {
    constraint_violation_msg =
        "qsort_s: array_size cannot be greater than RSIZE_MAX";
  } else if (elem_size > RSIZE_MAX) {
    constraint_violation_msg =
        "qsort_s: elem_size cannot be greater than RSIZE_MAX";
  } else if ((array_size != 0) && (array == 0 || compare == 0)) {
    constraint_violation_msg =
        "qsort_s: if array_size is not equal to zero, then neither array nor "
        "compare can be a null pointer";
  }
  if (constraint_violation_msg) {
    libc_global_constraint_handler(constraint_violation_msg, 0, ERRNO_T_FAIL);
    return ERRNO_T_FAIL;
  }

  internal::unstable_sort(array, array_size, elem_size, compare, context);
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
