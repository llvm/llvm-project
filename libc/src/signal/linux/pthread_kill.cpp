//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of pthread_kill.
///
//===----------------------------------------------------------------------===//

#include "src/signal/pthread_kill.h"

#include "hdr/errno_macros.h"
#include "hdr/signal_macros.h"
#include "hdr/types/pthread_t.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/__support/threads/thread.h"

namespace LIBC_NAMESPACE_DECL {

static_assert(sizeof(pthread_t) == sizeof(LIBC_NAMESPACE::Thread),
              "Mismatch between pthread_t and internal Thread.");

LLVM_LIBC_FUNCTION(int, pthread_kill, (pthread_t th, int sig)) {
  auto *thread = reinterpret_cast<Thread *>(&th);
  LIBC_CRASH_ON_NULLPTR(thread->attrib);

  // We can't delegate this check to the kernel since some of the code paths
  // don't go through the syscall.
  if (sig < 0 || sig >= NSIG)
    return EINVAL;

  auto res = thread->kill(sig);
  return res ? 0 : res.error();
}

} // namespace LIBC_NAMESPACE_DECL
