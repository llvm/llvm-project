//===-- Implementation header of fputs --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_FPUTS_H
#define LLVM_LIBC_SRC_STDIO_FPUTS_H

#include <stdio.h>

namespace __llvm_libc {

int fputs(const char *__restrict str, ::FILE *__restrict stream);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_FPUTS_H
