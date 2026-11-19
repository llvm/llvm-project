//===-- Linux implementation of the pthread_mutex_destroy function --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_mutex_destroy.h"

#include "hdr/errno_macros.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/mutex.h"
#include "src/__support/threads/mutex_common.h"

#include <pthread.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, pthread_mutex_destroy, (pthread_mutex_t * mutex)) {
  auto *m = reinterpret_cast<Mutex *>(mutex);
  return Mutex::destroy(m) == MutexError::BUSY ? EBUSY : 0;
}

} // namespace LIBC_NAMESPACE_DECL
