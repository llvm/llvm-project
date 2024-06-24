//===-- Linux implementation of the cnd_signal function -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/threads/cnd_signal.h"
#include "src/__support/common.h"
#include "src/__support/threads/CndVar.h"

#include <threads.h> // cnd_t, thrd_error, thrd_success

namespace LIBC_NAMESPACE {

static_assert(sizeof(CndVar) == sizeof(cnd_t));

LLVM_LIBC_FUNCTION(int, cnd_signal, (cnd_t * cond)) {
  CndVar *cndvar = reinterpret_cast<CndVar *>(cond);
  cndvar->notify_one();
  return thrd_success;
}

} // namespace LIBC_NAMESPACE
