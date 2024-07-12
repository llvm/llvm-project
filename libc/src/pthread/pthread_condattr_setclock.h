//===-- Implementation header for pthread_condattr_setclock -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_PTHREAD_PTHREAD_CONDATTR_SETCLOCK_H
#define LLVM_LIBC_SRC_PTHREAD_PTHREAD_CONDATTR_SETCLOCK_H

#include <pthread.h>
#include <sys/types.h> // clockid_t

namespace LIBC_NAMESPACE {

int pthread_condattr_setclock(pthread_condattr_t *attr, clockid_t clock);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_PTHREAD_PTHREAD_CONDATTR_SETCLOCK_H
