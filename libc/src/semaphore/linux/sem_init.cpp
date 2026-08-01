//===-- Linux implementation of sem_init ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/semaphore/sem_init.h"

#include "hdr/errno_macros.h"
#include "hdr/limits_macros.h"
#include "hdr/types/sem_t.h"
#include "src/__support/CPP/new.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/semaphore/linux/semaphore.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, sem_init,
                   (sem_t * sem, int pshared, unsigned int value)) {
  if (value > SEM_VALUE_MAX) {
    libc_errno = EINVAL;
    return -1;
  }

  new (sem) Semaphore(value, /*shared=*/pshared != 0);
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
