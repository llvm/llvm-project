//===-- Implementation header for write -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_UNISTD_WRITE_H
#define LLVM_LIBC_SRC_UNISTD_WRITE_H

#include "include/unistd.h"
#include <stddef.h>

namespace __llvm_libc {

ssize_t write(int fd, const void *buf, size_t count);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_UNISTD_WRITE_H
