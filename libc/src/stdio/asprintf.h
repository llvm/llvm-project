//===-- Implementation header of asprintf ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_ASPRINTF_H
#define LLVM_LIBC_SRC_STDIO_ASPRINTF_H

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE {

int asprintf(char **__restrict s, const char *format, ...);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STDIO_ASPRINTF_H
