//===-- Implementation of sem_destroy ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/semaphore/sem_destroy.h"

#include "src/semaphore/posix_semaphore.h"

#include "hdr/errno_macros.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, sem_destroy, (sem_t * sem)) {
  if (!sem_utils::is_valid(sem)) {
    libc_errno = EINVAL;
    return -1;
  }

  sem_utils::invalidate(sem);
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
