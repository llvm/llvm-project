//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for pthread_mutex_timedlock.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_PTHREAD_PTHREAD_MUTEX_TIMEDLOCK_H
#define LLVM_LIBC_SRC_PTHREAD_PTHREAD_MUTEX_TIMEDLOCK_H

#include "hdr/types/struct_timespec.h"
#include "src/__support/macros/config.h"
#include <pthread.h>

namespace LIBC_NAMESPACE_DECL {

int pthread_mutex_timedlock(pthread_mutex_t *__restrict mutex,
                            const struct timespec *__restrict abstime);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_PTHREAD_PTHREAD_MUTEX_TIMEDLOCK_H
