//===-- Implementation header for read --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_UNISTD_READ_H
#define LLVM_LIBC_SRC_UNISTD_READ_H

#include <unistd.h>

namespace __llvm_libc {

ssize_t read(int fd, void *buf, size_t count);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_UNISTD_READ_H
