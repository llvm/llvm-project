//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the pthread_getattr_np function (GNU extension).
///
//===----------------------------------------------------------------------===//

#include "src/pthread/pthread_getattr_np.h"
#include "hdr/pthread_macros.h"
#include "hdr/types/pthread_attr_t.h"
#include "hdr/types/pthread_t.h"
#include "src/__support/CPP/atomic.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/__support/threads/thread.h"

namespace LIBC_NAMESPACE_DECL {

static_assert(sizeof(pthread_t) == sizeof(LIBC_NAMESPACE::Thread),
              "Mismatch between pthread_t and internal Thread.");

LLVM_LIBC_FUNCTION(int, pthread_getattr_np,
                   (pthread_t th, pthread_attr_t *attr)) {
  LIBC_CRASH_ON_NULLPTR(attr);
  auto *thread = reinterpret_cast<Thread *>(&th);

  uint32_t detach_state =
      thread->attrib->detach_state.load(cpp::MemoryOrder::RELAXED);
  attr->__detachstate =
      (detach_state == static_cast<uint32_t>(DetachState::DETACHED))
          ? PTHREAD_CREATE_DETACHED
          : PTHREAD_CREATE_JOINABLE;
  attr->__stack = thread->attrib->stack;
  attr->__stacksize = thread->attrib->stacksize;
  attr->__guardsize = thread->attrib->guardsize;
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
