//===-- Implementation header for fchmodat ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_STAT_FCHMODAT_H
#define LLVM_LIBC_SRC_SYS_STAT_FCHMODAT_H

#include <sys/stat.h>

namespace __llvm_libc {

int fchmodat(int dirfd, const char *path, mode_t mode, int flags);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SYS_STAT_FCHMODAT_H
