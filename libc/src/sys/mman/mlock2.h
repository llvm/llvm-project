//===-- Implementation header for mlock2 function ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_MMAN_MLOCK2_H
#define LLVM_LIBC_SRC_SYS_MMAN_MLOCK2_H

#include <sys/mman.h> // For size_t

namespace LIBC_NAMESPACE {

int mlock2(const void *addr, size_t len, unsigned int flags);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_SYS_MMAN_MLOCK2_H
