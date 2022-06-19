//===-- Linux implementation of the cnd_wait function ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CndVar.h"

#include "src/__support/common.h"
#include "src/__support/threads/mutex.h"
#include "src/threads/cnd_wait.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, cnd_wait, (cnd_t * cond, mtx_t *mtx)) {
  CndVar *cndvar = reinterpret_cast<CndVar *>(cond);
  Mutex *mutex = reinterpret_cast<Mutex *>(mtx);
  return cndvar->wait(mutex);
}

} // namespace __llvm_libc
