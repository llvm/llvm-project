//===-- Implementation of sem_init ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/semaphore/sem_init.h"

#include "src/semaphore/posix_semaphore.h"

#include "hdr/errno_macros.h"
#include "hdr/limits_macros.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, sem_init,
                   (sem_t * sem, int pshared, unsigned int value)) {
  if (sem == nullptr || value > SEM_VALUE_MAX) {
    libc_errno = EINVAL;
    return -1;
  }

  (void)pshared;
  reinterpret_cast<Semaphore *>(sem)->init(value);
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
