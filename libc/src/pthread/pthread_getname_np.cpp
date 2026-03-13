//===-- Linux implementation of the pthread_setname_np function -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_getname_np.h"

#include "src/__support/CPP/span.h"
#include "src/__support/CPP/stringstream.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/thread.h"

#include <pthread.h>
#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {

static_assert(sizeof(pthread_t) == sizeof(LIBC_NAMESPACE::Thread),
              "Mismatch between pthread_t and internal Thread.");

LLVM_LIBC_FUNCTION(int, pthread_getname_np,
                   (pthread_t th, char *buf, size_t len)) {
  auto *thread = reinterpret_cast<LIBC_NAMESPACE::Thread *>(&th);
  cpp::span<char> name_buf(buf, len);
  cpp::StringStream name_stream(name_buf);
  return thread->get_name(name_stream);
}

} // namespace LIBC_NAMESPACE_DECL
