//===-- Starting point for scanf --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_SCANF_CORE_SCANF_MAIN_H
#define LLVM_LIBC_SRC_STDIO_SCANF_CORE_SCANF_MAIN_H

#include "src/__support/arg_list.h"
#include "src/stdio/scanf_core/reader.h"

#include <stddef.h>

namespace __llvm_libc {
namespace scanf_core {

int scanf_main(Reader *reader, const char *__restrict str,
               internal::ArgList &args);

} // namespace scanf_core
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_SCANF_CORE_SCANF_MAIN_H
