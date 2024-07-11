//===-- Linux implementation of the cnd_broadcast function ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/threads/cnd_broadcast.h"
#include "src/__support/common.h"
#include "src/__support/threads/CndVar.h"

// TODO: https://github.com/llvm/llvm-project/issues/92968
#include <threads.h> // cnd_t, thrd_error, thrd_success

namespace LIBC_NAMESPACE {

static_assert(sizeof(CndVar) == sizeof(cnd_t));

LLVM_LIBC_FUNCTION(int, cnd_broadcast, (cnd_t * cond)) {
  CndVar *cndvar = reinterpret_cast<CndVar *>(cond);
  cndvar->broadcast();
  return thrd_success;
}

} // namespace LIBC_NAMESPACE
