//===-- Implementation of the pthread_getunique_np function ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_getunique_np.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"

#include <pthread.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, pthread_getunique_np,
                   (const pthread_t *__restrict thread,
                    pthread_id_np_t *__restrict id)) {
  LIBC_CRASH_ON_NULLPTR(id);
  // We assume that unique thread ID is an integer value of a pointer to TCB.
  *id = (thread == nullptr)
            ? 0
            : reinterpret_cast<pthread_id_np_t>(thread->__attrib);
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
