//===-- Linux implementation of the callonce function ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "futex_word.h"

#include "include/sys/syscall.h" // For syscall numbers.
#include "src/__support/CPP/atomic.h"
#include "src/__support/OSUtil/syscall.h" // For syscall functions.
#include "src/__support/threads/callonce.h"

#include <limits.h>
#include <linux/futex.h>

namespace __llvm_libc {

static constexpr FutexWordType NOT_CALLED = 0x0;
static constexpr FutexWordType START = 0x11;
static constexpr FutexWordType WAITING = 0x22;
static constexpr FutexWordType FINISH = 0x33;

int callonce(CallOnceFlag *flag, CallOnceCallback *func) {
  auto *futex_word = reinterpret_cast<cpp::Atomic<FutexWordType> *>(flag);

  FutexWordType not_called = NOT_CALLED;

  // The call_once call can return only after the called function |func|
  // returns. So, we use futexes to synchronize calls with the same flag value.
  if (futex_word->compare_exchange_strong(not_called, START)) {
    func();
    auto status = futex_word->exchange(FINISH);
    if (status == WAITING) {
      __llvm_libc::syscall(SYS_futex, &futex_word->val, FUTEX_WAKE_PRIVATE,
                           INT_MAX, // Wake all waiters.
                           0, 0, 0);
    }
    return 0;
  }

  FutexWordType status = START;
  if (futex_word->compare_exchange_strong(status, WAITING) ||
      status == WAITING) {
    __llvm_libc::syscall(SYS_futex, &futex_word->val, FUTEX_WAIT_PRIVATE,
                         WAITING, // Block only if status is still |WAITING|.
                         0, 0, 0);
  }

  return 0;
}

} // namespace __llvm_libc
