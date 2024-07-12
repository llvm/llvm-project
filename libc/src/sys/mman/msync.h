//===-- Implementation header for msync function ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_MMAN_MSYNC_H
#define LLVM_LIBC_SRC_SYS_MMAN_MSYNC_H

#include <sys/mman.h>
#include <sys/syscall.h>

namespace LIBC_NAMESPACE {

int msync(void *addr, size_t len, int flags);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_SYS_MMAN_MSYNC_H
