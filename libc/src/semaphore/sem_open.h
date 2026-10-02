//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for sem_open.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SEMAPHORE_SEM_OPEN_H
#define LLVM_LIBC_SRC_SEMAPHORE_SEM_OPEN_H

#include "hdr/types/sem_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

// POSIX leaves it implementation-defined whether repeated calls for the same
// name return the same handle. Each successful call here returns a unique
// handle, and every handle for a name refers to the same underlying semaphore.
// Each handle must be released with its own sem_close call.
sem_t *sem_open(const char *name, int oflag, ...);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SEMAPHORE_SEM_OPEN_H
