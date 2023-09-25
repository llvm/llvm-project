//===-- Implementation header for div ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_DIV_H
#define LLVM_LIBC_SRC_STDLIB_DIV_H

#include <stdlib.h>

namespace __llvm_libc {

div_t div(int x, int y);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDLIB_DIV_H
