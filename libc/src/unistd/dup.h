//===-- Implementation header for dup ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_UNISTD_DUP_H
#define LLVM_LIBC_SRC_UNISTD_DUP_H

#include <unistd.h>

namespace __llvm_libc {

int dup(int fd);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_UNISTD_DUP_H
