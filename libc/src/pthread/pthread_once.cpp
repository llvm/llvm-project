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

LLVM_LIBC_FUNCTION(int, pthread_once,
                   (pthread_once_t * flag, __pthread_once_func_t func)) {
  return callonce(reinterpret_cast<CallOnceFlag *>(flag), func);
}

} // namespace LIBC_NAMESPACE_DECL
