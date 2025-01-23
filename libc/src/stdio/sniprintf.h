//===-- Implementation header of sniprintf ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_SNIPRINTF_H
#define LLVM_LIBC_SRC_STDIO_SNIPRINTF_H

#include "src/__support/macros/config.h"
#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {

int sniprintf(char *__restrict buffer, size_t buffsz,
              const char *__restrict format, ...);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDIO_SNIPRINTF_H
