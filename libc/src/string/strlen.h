//===-- Implementation header for strlen ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_STRLEN_H
#define LLVM_LIBC_SRC_STRING_STRLEN_H

#include <string.h>

namespace __llvm_libc {

size_t strlen(const char *src);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STRING_STRLEN_H
