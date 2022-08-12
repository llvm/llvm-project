//===-- Implementation of the pthread_key_delete --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_key_delete.h"

#include "src/__support/common.h"
#include "src/__support/threads/thread.h"

#include <errno.h>
#include <pthread.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, pthread_key_delete, (pthread_key_t key)) {
  if (__llvm_libc::tss_key_delete(key))
    return 0;
  else
    return EINVAL;
}

} // namespace __llvm_libc
