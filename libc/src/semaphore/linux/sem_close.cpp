//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of sem_close.
///
//===----------------------------------------------------------------------===//

#include "src/semaphore/sem_close.h"

#include "hdr/errno_macros.h"
#include "hdr/types/sem_t.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/semaphore/linux/semaphore.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, sem_close, (sem_t * sem)) {
  LIBC_CRASH_ON_NULLPTR(sem);

  Semaphore *semaphore = reinterpret_cast<Semaphore *>(sem);
  if (!semaphore->is_valid()) {
    libc_errno = EINVAL;
    return -1;
  }

  if (int err = Semaphore::close(semaphore)) {
    libc_errno = err;
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
