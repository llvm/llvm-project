//===-- Implementation header for shm_open function -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_MMAN_SHM_OPEN_H
#define LLVM_LIBC_SRC_SYS_MMAN_SHM_OPEN_H

#include <llvm-libc-types/mode_t.h>

namespace LIBC_NAMESPACE {

int shm_open(const char *name, int oflag, mode_t mode);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_SYS_MMAN_SHM_OPEN_H
