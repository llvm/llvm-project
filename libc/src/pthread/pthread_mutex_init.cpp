//===-- Linux implementation of the pthread_mutex_init function -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_mutex_init.h"
#include "pthread_mutexattr.h"

#include "src/__support/common.h"
#include "src/__support/threads/mutex.h"

#include <errno.h>
#include <pthread.h>

namespace __llvm_libc {

static_assert(sizeof(Mutex) <= sizeof(pthread_mutex_t),
              "The public pthread_mutex_t type cannot accommodate the internal "
              "mutex type.");

LLVM_LIBC_FUNCTION(int, pthread_mutex_init,
                   (pthread_mutex_t * m,
                    const pthread_mutexattr_t *__restrict attr)) {
  auto mutexattr = attr == nullptr ? DEFAULT_MUTEXATTR : *attr;
  auto err =
      Mutex::init(reinterpret_cast<Mutex *>(m), false,
                  get_mutexattr_type(mutexattr) & PTHREAD_MUTEX_RECURSIVE,
                  get_mutexattr_robust(mutexattr) & PTHREAD_MUTEX_ROBUST);
  return err == MutexError::NONE ? 0 : EAGAIN;
}

} // namespace __llvm_libc
