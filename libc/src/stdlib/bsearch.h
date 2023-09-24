//===-- Implementation header for bsearch -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_BSEARCH_H
#define LLVM_LIBC_SRC_STDLIB_BSEARCH_H

#include <stdlib.h>

namespace __llvm_libc {

void *bsearch(const void *key, const void *array, size_t array_size,
              size_t elem_size, int (*compare)(const void *, const void *));

} // namespace __llvm_libc

#endif //LLVM_LIBC_SRC_STDLIB_BSEARCH_H
