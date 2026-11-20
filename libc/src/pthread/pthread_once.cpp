//===-- Linux implementation of the pthread_once function -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_once.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/callonce.h"

#include <pthread.h> // For pthread_once_t and __pthread_once_func_t definitions.

namespace LIBC_NAMESPACE_DECL {

// pthread_once_t is an int; the flag is the same four bytes viewed as the
// futex word call_once uses.
static_assert(sizeof(pthread_once_t) == sizeof(CallOnceFlag) &&
                  alignof(pthread_once_t) >= alignof(CallOnceFlag),
              "pthread_once_t must be able to hold a CallOnceFlag");

LLVM_LIBC_FUNCTION(int, pthread_once,
                   (pthread_once_t * flag, __pthread_once_func_t func)) {
  return callonce(reinterpret_cast<CallOnceFlag *>(flag), func);
}

} // namespace LIBC_NAMESPACE_DECL
