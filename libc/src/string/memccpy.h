//===-- Implementation header for memccpy -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMCCPY_H
#define LLVM_LIBC_SRC_STRING_MEMCCPY_H

#include <stddef.h>

namespace __llvm_libc {

void *memccpy(void *__restrict dest, const void *__restrict src, int c,
              size_t count);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STRING_MEMCCPY_H
