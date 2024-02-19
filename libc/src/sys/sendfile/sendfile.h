//===-- Implementation header for sendfile ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_SENDFILE_SENDFILE_H
#define LLVM_LIBC_SRC_SYS_SENDFILE_SENDFILE_H

#include <sys/sendfile.h>

namespace LIBC_NAMESPACE {

ssize_t sendfile(int, int, off_t *, size_t);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_SYS_SENDFILE_SENDFILE_H
