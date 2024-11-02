//===-- Internal implementation header of vfprintf --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_PRINTF_CORE_VFPRINTF_INTERNAL_H
#define LLVM_LIBC_SRC_STDIO_PRINTF_CORE_VFPRINTF_INTERNAL_H

#include "src/__support/arg_list.h"

#include <stdio.h>

namespace __llvm_libc {
namespace printf_core {

int vfprintf_internal(::FILE *__restrict stream, const char *__restrict format,
                      internal::ArgList &args);
} // namespace printf_core
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_CORE_VFPRINTF_INTERNAL_H
