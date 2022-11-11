//===-- Implementation header for sendfile ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_STAT_SENDFILE_H
#define LLVM_LIBC_SRC_SYS_STAT_SENDFILE_H

#include <sys/sendfile.h>

namespace __llvm_libc {

ssize_t sendfile(int, int, off_t *, size_t);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SYS_STAT_SENDFILE_H
