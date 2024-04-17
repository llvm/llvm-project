//===-- Linux implementation of the pthread_cond_wait function ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_cond_wait.h"

#include "include/llvm-libc-macros/generic-error-number-macros.h" // EINVAL
#include "src/__support/common.h"
#include "src/threads/linux/CndVar.h"

#include <pthread.h> // pthread_cond_t, pthread_mutex_t
// TODO: https://github.com/llvm/llvm-project/issues/88580
#include <threads.h> // thrd_success

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, pthread_cond_wait,
                   (pthread_cond_t * cond, pthread_mutex_t *mutex)) {
  if (!cond || !mutex)
    return EINVAL;

  auto *C = reinterpret_cast<CndVar *>(cond);
  auto *M = reinterpret_cast<Mutex *>(mutex);
  int ret = C->wait(M);

  if (ret == thrd_success)
    return 0;

  // TODO translate error codes
  return -1;
}

} // namespace LIBC_NAMESPACE
