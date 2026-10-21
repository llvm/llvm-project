//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the pthread_getstack_np function (LLVM-libc extension).
///
//===----------------------------------------------------------------------===//

#include "src/pthread/pthread_getstack_np.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/__support/threads/thread.h"

namespace LIBC_NAMESPACE_DECL {

static_assert(sizeof(pthread_t) == sizeof(LIBC_NAMESPACE::Thread),
              "Mismatch between pthread_t and internal Thread.");

LLVM_LIBC_FUNCTION(int, pthread_getstack_np,
                   (pthread_t th, void **__restrict stackaddr,
                    size_t *__restrict stacksize)) {
  LIBC_CRASH_ON_NULLPTR(stackaddr);
  LIBC_CRASH_ON_NULLPTR(stacksize);
  auto *thread = reinterpret_cast<Thread *>(&th);
  *stackaddr = thread->attrib->stack;
  *stacksize = thread->attrib->stacksize;
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
