//===-- Implementation of the thrd_current function -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/threads/thrd_current.h"
#include "src/__support/common.h"
#include "src/__support/threads/thread.h"

#include <threads.h> // For thrd_* type definitions.

namespace __llvm_libc {

static_assert(sizeof(thrd_t) == sizeof(__llvm_libc::Thread),
              "Mismatch between thrd_t and internal Thread.");

LLVM_LIBC_FUNCTION(thrd_t, thrd_current, ()) {
  thrd_t th;
  th.__attrib = self.attrib;
  return th;
}

} // namespace __llvm_libc
