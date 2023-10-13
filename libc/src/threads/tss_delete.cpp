//===-- Implementation of the tss_delete ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "tss_delete.h"

#include "src/__support/common.h"
#include "src/__support/threads/thread.h"

#include <threads.h>

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(void, tss_delete, (tss_t key)) {
  LIBC_NAMESPACE::tss_key_delete(key);
}

} // namespace LIBC_NAMESPACE
