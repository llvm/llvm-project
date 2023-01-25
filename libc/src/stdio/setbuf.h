//===-- Implementation header of setbuf -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_SETBUF_H
#define LLVM_LIBC_SRC_STDIO_SETBUF_H

#include <stdio.h>

namespace __llvm_libc {

void setbuf(::FILE *__restrict stream, char *__restrict buf);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_SETBUF_H
