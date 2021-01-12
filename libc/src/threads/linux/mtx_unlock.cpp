//===-- Linux implementation of the mtx_unlock function -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/threads/mtx_unlock.h"
#include "config/linux/syscall.h" // For syscall functions.
#include "include/sys/syscall.h"  // For syscall numbers.
#include "include/threads.h"      // For mtx_t definition.
#include "src/__support/common.h"
#include "src/threads/linux/thread_utils.h"

#include <linux/futex.h> // For futex operations.
#include <stdatomic.h>   // for atomic_compare_exchange_strong.

namespace __llvm_libc {

// The implementation currently handles only plain mutexes.
LLVM_LIBC_FUNCTION(int, mtx_unlock, (mtx_t * mutex)) {
  FutexData *futex_word = reinterpret_cast<FutexData *>(mutex->__internal_data);
  while (true) {
    uint32_t mutex_status = MS_Waiting;
    if (atomic_compare_exchange_strong(futex_word, &mutex_status, MS_Free)) {
      // If any thread is waiting to be woken up, then do it.
      __llvm_libc::syscall(SYS_futex, futex_word, FUTEX_WAKE_PRIVATE, 1, 0, 0,
                           0);
      return thrd_success;
    }

    if (mutex_status == MS_Locked) {
      // If nobody was waiting at this point, just free it.
      if (atomic_compare_exchange_strong(futex_word, &mutex_status, MS_Free))
        return thrd_success;
    } else {
      // This can happen, for example if some thread tries to unlock an already
      // free mutex.
      return thrd_error;
    }
  }
}

} // namespace __llvm_libc
