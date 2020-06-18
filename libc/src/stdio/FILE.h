//===-- Internal definition of FILE -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_FILE_H
#define LLVM_LIBC_SRC_STDIO_FILE_H

#include "include/threads.h"
#include <stddef.h>

namespace __llvm_libc {

struct FILE {
  mtx_t lock;

  using write_function_t = size_t(FILE *, const char *, size_t);

  write_function_t *write;
};

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_FILE_H
