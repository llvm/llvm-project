//===-- Linux implementation of sem_open ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/semaphore/sem_open.h"

#include "hdr/fcntl_macros.h"
#include "hdr/semaphore_macros.h"
#include "hdr/types/mode_t.h"
#include "hdr/types/sem_t.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/semaphore/linux/semaphore.h"

#include <stdarg.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(sem_t *, sem_open, (const char *name, int oflag, ...)) {
  mode_t mode = 0;
  unsigned int value = 0;

  if (oflag & O_CREAT) {
    va_list varargs;
    va_start(varargs, oflag);
    mode = va_arg(varargs, mode_t);
    value = va_arg(varargs, unsigned int);
    va_end(varargs);
  }

  auto sem_or = Semaphore::open(name, oflag, mode, value);
  if (!sem_or.has_value()) {
    libc_errno = sem_or.error();
    return SEM_FAILED;
  }

  return reinterpret_cast<sem_t *>(sem_or.value());
}

} // namespace LIBC_NAMESPACE_DECL
