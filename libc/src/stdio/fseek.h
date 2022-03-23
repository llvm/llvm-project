//===-- Implementation header of fseek --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_FSEEK_H
#define LLVM_LIBC_SRC_STDIO_FSEEK_H

#include <stdio.h>

namespace __llvm_libc {

int fseek(::FILE *stream, long offset, int whence);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_FSEEK_H
