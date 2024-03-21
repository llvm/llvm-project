//===-- Implementation header for lseek -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_UNISTD_LSEEK_H
#define LLVM_LIBC_SRC_UNISTD_LSEEK_H

#include "include/llvm-libc-types/off_t.h"

namespace LIBC_NAMESPACE {

off_t lseek(int fd, off_t offset, int whence);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_UNISTD_LSEEK_H
