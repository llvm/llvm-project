//===-- Linux implementation of the tss_get function ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "tss_get.h"

#include "src/__support/common.h"
#include "src/__support/threads/thread.h"

#include <threads.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(void *, tss_get, (tss_t key)) { return get_tss_value(key); }

} // namespace __llvm_libc
