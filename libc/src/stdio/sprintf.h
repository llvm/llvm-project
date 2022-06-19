//===-- Implementation header of sprintf ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_SPRINTF_H
#define LLVM_LIBC_SRC_STDIO_SPRINTF_H

namespace __llvm_libc {

int sprintf(char *__restrict buffer, const char *__restrict format, ...);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_SPRINTF_H
