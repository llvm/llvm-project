//===-- Implementation of sem_getvalue -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/semaphore/sem_getvalue.h"

#include "src/semaphore/posix_semaphore.h"

#include "hdr/errno_macros.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, sem_getvalue,
                   (sem_t *__restrict sem, int *__restrict sval)) {
  if (!sem_utils::is_valid(sem) || sval == nullptr) {
    libc_errno = EINVAL;
    return -1;
  }

  // get value is informational but not a synchronization op
  // RELAXED ordering is enough
  *sval =
      static_cast<int>(sem_utils::value(sem).load(cpp::MemoryOrder::RELAXED));
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
