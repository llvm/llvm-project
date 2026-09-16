//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of pthread_attr_init.
///
//===----------------------------------------------------------------------===//

#include "src/pthread/pthread_attr_init.h"
#include "hdr/pthread_macros.h"
#include "hdr/sched_macros.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/__support/threads/thread.h" // For thread::DEFAULT_*

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, pthread_attr_init, (pthread_attr_t * attr)) {
  LIBC_CRASH_ON_NULLPTR(attr);

  *attr = pthread_attr_t{
      PTHREAD_CREATE_JOINABLE,   // Not detached
      PTHREAD_INHERIT_SCHED,     // Default inherit scheduler
      SCHED_OTHER,               // Default scheduling policy
      {},                        // Default scheduling parameters
      nullptr,                   // Let the thread manage its stack
      Thread::DEFAULT_STACKSIZE, // stack size.
      Thread::DEFAULT_GUARDSIZE, // Default page size for the guard size.
  };
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
