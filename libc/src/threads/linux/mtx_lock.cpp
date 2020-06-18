//===-- Linux implementation of the mtx_lock function ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "config/linux/syscall.h" // For syscall functions.
#include "include/sys/syscall.h"  // For syscall numbers.
#include "include/threads.h"      // For mtx_t definition.
#include "src/__support/common.h"
#include "src/threads/linux/thread_utils.h"

#include <linux/futex.h> // For futex operations.
#include <stdatomic.h>   // For atomic_compare_exchange_strong.

namespace __llvm_libc {

// The implementation currently handles only plain mutexes.
int LLVM_LIBC_ENTRYPOINT(mtx_lock)(mtx_t *mutex) {
  FutexData *futex_data = reinterpret_cast<FutexData *>(mutex->__internal_data);
  while (true) {
    uint32_t mutex_status = MS_Free;
    uint32_t locked_status = MS_Locked;

    if (atomic_compare_exchange_strong(futex_data, &mutex_status, MS_Locked))
      return thrd_success;

    switch (mutex_status) {
    case MS_Waiting:
      // If other threads are waiting already, then join them. Note that the
      // futex syscall will block if the futex data is still `MS_Waiting` (the
      // 4th argument to the syscall function below.)
      __llvm_libc::syscall(SYS_futex, futex_data, FUTEX_WAIT_PRIVATE,
                           MS_Waiting, 0, 0, 0);
      // Once woken up/unblocked, try everything all over.
      continue;
    case MS_Locked:
      // Mutex has been locked by another thread so set the status to
      // MS_Waiting.
      if (atomic_compare_exchange_strong(futex_data, &locked_status,
                                         MS_Waiting)) {
        // If we are able to set the futex data to `MS_Waiting`, then we will
        // wait for the futex to be woken up. Note again that the following
        // syscall will block only if the futex data is still `MS_Waiting`.
        __llvm_libc::syscall(SYS_futex, futex_data, FUTEX_WAIT_PRIVATE,
                             MS_Waiting, 0, 0, 0);
      }
      continue;
    case MS_Free:
      // If it was MS_Free, we shouldn't be here at all.
      [[clang::fallthrough]];
    default:
      // Mutex status cannot be anything else. So control should not reach
      // here at all.
      return thrd_error;
    }
  }
}

} // namespace __llvm_libc
