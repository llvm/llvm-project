//===-- Linux implementation of the pthread_setname_np function -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_setname_np.h"

#include "src/__support/CPP/string_view.h"
#include "src/__support/common.h"
#include "src/__support/threads/thread.h"

#include <pthread.h>

namespace LIBC_NAMESPACE {

static_assert(sizeof(pthread_t) == sizeof(LIBC_NAMESPACE::Thread),
              "Mismatch between pthread_t and internal Thread.");

LLVM_LIBC_FUNCTION(int, pthread_setname_np, (pthread_t th, const char *name)) {
  auto *thread = reinterpret_cast<LIBC_NAMESPACE::Thread *>(&th);
  return thread->set_name(cpp::string_view(name));
}

} // namespace LIBC_NAMESPACE
