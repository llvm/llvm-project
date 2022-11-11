//===-- Implementation header for fstat -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_STAT_FSTAT_H
#define LLVM_LIBC_SRC_SYS_STAT_FSTAT_H

#include <sys/stat.h>

namespace __llvm_libc {

int fstat(int fd, struct stat *statbuf);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SYS_STAT_FSTAT_H
