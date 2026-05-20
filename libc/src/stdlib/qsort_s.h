//===-- Implementation header for qsort_s------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_QSORT_S_H
#define LLVM_LIBC_SRC_STDLIB_QSORT_S_H

#include "hdr/types/errno_t.h"
#include "hdr/types/rsize_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

errno_t qsort_s(void *array, rsize_t array_size, rsize_t elem_size,
                int (*compare)(const void *, const void *, void *),
                void *context);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDLIB_QSORT_H
