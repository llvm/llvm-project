//===-- Implementation header for dup2 --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_UNISTD_DUP2_H
#define LLVM_LIBC_SRC_UNISTD_DUP2_H

#include <unistd.h>

namespace __llvm_libc {

int dup2(int oldfd, int newfd);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_UNISTD_DUP2_H
