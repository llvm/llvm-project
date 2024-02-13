//===-- Linux implementation of the mtx_unlock function -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/threads/mtx_unlock.h"
#include "src/__support/common.h"
#include "src/__support/threads/mutex.h"

#include <threads.h> // For mtx_t definition.

namespace LIBC_NAMESPACE {

// The implementation currently handles only plain mutexes.
LLVM_LIBC_FUNCTION(int, mtx_unlock, (mtx_t * mutex)) {
  auto *m = reinterpret_cast<Mutex *>(mutex);
  auto err = m->unlock();
  return err == MutexError::NONE ? thrd_success : thrd_error;
}

} // namespace LIBC_NAMESPACE
