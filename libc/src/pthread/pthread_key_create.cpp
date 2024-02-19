//===-- Implementation of the pthread_key_create --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_key_create.h"

#include "src/__support/common.h"
#include "src/__support/threads/thread.h"

#include <errno.h>
#include <pthread.h>

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, pthread_key_create,
                   (pthread_key_t * key, __pthread_tss_dtor_t dtor)) {
  auto k = LIBC_NAMESPACE::new_tss_key(dtor);
  if (!k)
    return EINVAL;
  *key = *k;
  return 0;
}

} // namespace LIBC_NAMESPACE
