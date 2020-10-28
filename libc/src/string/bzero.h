//===-- Implementation header for bzero -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_BZERO_H
#define LLVM_LIBC_SRC_STRING_BZERO_H

#include <stddef.h> // size_t

namespace __llvm_libc {

void bzero(void *ptr, size_t count);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STRING_BZERO_H
