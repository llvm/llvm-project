//===-- Implementation header for sem_open function -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SEMAPHORE_SEM_OPEN_H
#define LLVM_LIBC_SRC_SEMAPHORE_SEM_OPEN_H

#include "hdr/types/sem_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

sem_t *sem_open(const char *name, int oflag, ...);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SEMAPHORE_SEM_OPEN_H
