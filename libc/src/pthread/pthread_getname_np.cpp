//===-- Linux implementation of the pthread_setname_np function -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_getname_np.h"

#include "src/__support/CPP/ArrayRef.h"
#include "src/__support/CPP/StringView.h"
#include "src/__support/common.h"
#include "src/__support/threads/thread.h"

#include <pthread.h>
#include <stddef.h>

namespace __llvm_libc {

static_assert(sizeof(pthread_t) == sizeof(__llvm_libc::Thread),
              "Mismatch between pthread_t and internal Thread.");

LLVM_LIBC_FUNCTION(int, pthread_getname_np,
                   (pthread_t th, char *buf, size_t len)) {
  auto *thread = reinterpret_cast<__llvm_libc::Thread *>(&th);
  cpp::MutableArrayRef<char> name_buf(buf, len);
  cpp::StringStream name_stream(name_buf);
  return thread->get_name(name_stream);
}

} // namespace __llvm_libc
