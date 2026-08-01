//===-- Linux implementation of sem_destroy -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/semaphore/sem_destroy.h"

#include "hdr/errno_macros.h"
#include "hdr/types/sem_t.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/semaphore/linux/semaphore.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, sem_destroy, (sem_t * sem)) {
  Semaphore *semaphore = reinterpret_cast<Semaphore *>(sem);
  if (!semaphore->is_valid()) {
    libc_errno = EINVAL;
    return -1;
  }

  semaphore->destroy();
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
