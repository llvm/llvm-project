//===-- Implementation header for pread -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_UNISTD_PREAD_H
#define LLVM_LIBC_SRC_UNISTD_PREAD_H

#include "include/llvm-libc-types/off_t.h"
#include "include/llvm-libc-types/size_t.h"
#include "include/llvm-libc-types/ssize_t.h"

namespace LIBC_NAMESPACE {

ssize_t pread(int fd, void *buf, size_t count, off_t offset);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_UNISTD_PREAD_H
