//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the pthread_getcpuclockid function.
///
//===----------------------------------------------------------------------===//

#include "src/pthread/pthread_getcpuclockid.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/__support/threads/thread.h"

namespace LIBC_NAMESPACE_DECL {

static_assert(sizeof(pthread_t) == sizeof(LIBC_NAMESPACE::Thread),
              "Mismatch between pthread_t and internal Thread.");

LLVM_LIBC_FUNCTION(int, pthread_getcpuclockid,
                   (pthread_t th, clockid_t *clock_id)) {
  LIBC_CRASH_ON_NULLPTR(clock_id);
  auto *thread = reinterpret_cast<Thread *>(&th);
  LIBC_CRASH_ON_NULLPTR(thread->attrib);
  *clock_id = thread->getcpuclockid();
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
