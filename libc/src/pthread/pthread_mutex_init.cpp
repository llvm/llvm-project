//===-- Linux implementation of the pthread_mutex_init function -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_mutex_init.h"
#include "pthread_mutexattr.h"

#include "src/__support/CPP/new.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/mutex.h"

#include <pthread.h>

namespace LIBC_NAMESPACE_DECL {

static_assert(sizeof(Mutex) == sizeof(pthread_mutex_t) &&
                  alignof(Mutex) == alignof(pthread_mutex_t),
              "The public pthread_mutex_t type must exactly match the internal "
              "mutex type.");

LLVM_LIBC_FUNCTION(int, pthread_mutex_init,
                   (pthread_mutex_t * m,
                    const pthread_mutexattr_t *__restrict attr)) {
  auto mutexattr = attr == nullptr ? DEFAULT_MUTEXATTR : *attr;
  bool is_recursive = false;
  switch (get_mutexattr_type(mutexattr)) {
  case PTHREAD_MUTEX_NORMAL:
  case PTHREAD_MUTEX_ERRORCHECK:
    break;
  case PTHREAD_MUTEX_RECURSIVE:
    is_recursive = true;
    break;
  }

  bool is_robust = false;
  if (get_mutexattr_robust(mutexattr) == PTHREAD_MUTEX_ROBUST)
    is_robust = true;

  bool is_pshared = get_mutexattr_pshared(mutexattr) == PTHREAD_PROCESS_SHARED;

  new (m)
      Mutex(/*is_priority_inherit=*/false, is_recursive, is_robust, is_pshared);
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
