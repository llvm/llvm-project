//===-- Linux implementation of the pthread_rwlock_init function ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/pthread/pthread_rwlock_init.h"

#include "src/__support/CPP/new.h"
#include "src/__support/common.h"
#include "src/__support/libc_assert.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/linux/rwlock.h"

#include <pthread.h>

namespace LIBC_NAMESPACE_DECL {

static_assert(
    sizeof(RwLock) == sizeof(pthread_rwlock_t) &&
        alignof(RwLock) == alignof(pthread_rwlock_t),
    "The public pthread_rwlock_t type must be of the same size and alignment "
    "as the internal rwlock type.");

LLVM_LIBC_FUNCTION(int, pthread_rwlock_init,
                   (pthread_rwlock_t * rwlock,
                    const pthread_rwlockattr_t *__restrict attr)) {
  pthread_rwlockattr_t rwlockattr{
      /*pshared=*/PTHREAD_PROCESS_PRIVATE,
      /*pref*/ PTHREAD_RWLOCK_PREFER_READER_NP,
  };
  // POSIX does not specify this check, so we add an assertion to catch it.
  LIBC_ASSERT(rwlock && "rwlock is null");
  if (attr)
    rwlockattr = *attr;

  // PTHREAD_RWLOCK_PREFER_WRITER_NP is not supported.
  rwlock::Role preference;
  switch (rwlockattr.pref) {
  case PTHREAD_RWLOCK_PREFER_READER_NP:
    preference = rwlock::Role::Reader;
    break;
  case PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP:
    preference = rwlock::Role::Writer;
    break;
  default:
    return EINVAL;
  }
  bool is_pshared;
  switch (rwlockattr.pshared) {
  case PTHREAD_PROCESS_PRIVATE:
    is_pshared = false;
    break;
  case PTHREAD_PROCESS_SHARED:
    is_pshared = true;
    break;
  default:
    return EINVAL;
  }

  new (rwlock) RwLock(preference, is_pshared);
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
