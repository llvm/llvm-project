//===-- Linux implementation of the tss_set function ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "tss_set.h"

#include "src/__support/common.h"
#include "src/__support/threads/thread.h"

#include <threads.h>

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, tss_set, (tss_t key, void *data)) {
  if (set_tss_value(key, data))
    return thrd_success;
  else
    return thrd_error;
}

} // namespace LIBC_NAMESPACE
