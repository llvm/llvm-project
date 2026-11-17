//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for sem_close.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SEMAPHORE_SEM_CLOSE_H
#define LLVM_LIBC_SRC_SEMAPHORE_SEM_CLOSE_H

#include "hdr/types/sem_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int sem_close(sem_t *sem);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SEMAPHORE_SEM_CLOSE_H
