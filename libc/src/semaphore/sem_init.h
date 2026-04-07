//===-- Implementation header for sem_init ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SEMAPHORE_SEM_INIT_H
#define LLVM_LIBC_SRC_SEMAPHORE_SEM_INIT_H

#include "hdr/types/sem_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int sem_init(sem_t *sem, int pshared, unsigned int value);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SEMAPHORE_SEM_INIT_H
