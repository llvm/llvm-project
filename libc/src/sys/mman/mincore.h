//===-- Implementation header for mincore function --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_MMAN_MINCORE_H
#define LLVM_LIBC_SRC_SYS_MMAN_MINCORE_H

#include <sys/mman.h> // For size_t

namespace LIBC_NAMESPACE {

int mincore(void *addr, size_t len, unsigned char *vec);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_SYS_MMAN_MINCORE_H
