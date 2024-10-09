//===-- Implementation header of vasprintf ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_VASPRINTF_H
#define LLVM_LIBC_SRC_STDIO_VASPRINTF_H

#include <stdarg.h>

namespace LIBC_NAMESPACE {

int vasprintf(char **__restrict s, const char *format, va_list vlist);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STDIO_VASPRINTF_H
