//===-- Implementation header for memmove -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMMOVE_H
#define LLVM_LIBC_SRC_STRING_MEMMOVE_H

#include <stddef.h> // size_t

namespace __llvm_libc {

void *memmove(void *dest, const void *src, size_t count);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STRING_MEMMOVE_H
