//===-- Implementation header for strncpy -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_STRNCPY_H
#define LLVM_LIBC_SRC_STRING_STRNCPY_H

#include <stddef.h>

namespace __llvm_libc {

char *strncpy(char *__restrict dest, const char *__restrict src, size_t n);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STRING_STRNCPY_H
