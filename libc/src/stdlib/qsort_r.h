//===-- Implementation header for qsort_r -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_QSORT_R_H
#define LLVM_LIBC_SRC_STDLIB_QSORT_R_H

#include "hdr/types/size_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

// This qsort_r uses the POSIX 1003.1-2024 argument ordering instead of the
// historical BSD argument ordering (which put arg before the function pointer).
// https://www.austingroupbugs.net/view.php?id=900

void qsort_r(void *array, size_t array_size, size_t elem_size,
             int (*compare)(const void *, const void *, void *), void *arg);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDLIB_QSORT_R_H
