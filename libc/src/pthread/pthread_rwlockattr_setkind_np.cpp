//===-- Implementation of the pthread_rwlockattr_setkind_np ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_rwlockattr_setkind_np.h"

#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

#include <pthread.h> // pthread_rwlockattr_t

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, pthread_rwlockattr_setkind_np,
                   (pthread_rwlockattr_t * attr, int pref)) {

  if (pref != PTHREAD_RWLOCK_PREFER_READER_NP &&
      pref != PTHREAD_RWLOCK_PREFER_WRITER_NP &&
      pref != PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP)
    return EINVAL;

  attr->pref = pref;
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
