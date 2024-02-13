//===-- Implementation of the tss_create ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "tss_create.h"

#include "src/__support/common.h"
#include "src/__support/threads/thread.h"

#include <threads.h>

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, tss_create, (tss_t * key, tss_dtor_t dtor)) {
  auto k = LIBC_NAMESPACE::new_tss_key(dtor);
  if (!k)
    return thrd_error;
  *key = *k;
  return thrd_success;
}

} // namespace LIBC_NAMESPACE
