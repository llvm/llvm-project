//===-- Implementation of pthread_cond_signal -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_cond_signal.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/__support/threads/CndVar.h"

namespace LIBC_NAMESPACE_DECL {

static_assert(
    sizeof(CndVar) == sizeof(pthread_cond_t) &&
        alignof(CndVar) == alignof(pthread_cond_t),
    "The public pthread_cond_t type must be of the same size and alignment "
    "as the internal condition variable type.");

LLVM_LIBC_FUNCTION(int, pthread_cond_signal, (pthread_cond_t * cond)) {
  LIBC_CRASH_ON_NULLPTR(cond);
  // TODO: use cpp:start_lifetime_as once
  // https://github.com/llvm/llvm-project/pull/193326 is merged
  auto *cndvar = reinterpret_cast<CndVar *>(cond);
  cndvar->notify_one();
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
