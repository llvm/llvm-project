//===-- Linux implementation of the call_once function --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Futex.h"

#include "src/__support/CPP/atomic.h"
#include "src/__support/OSUtil/syscall.h" // For syscall functions.
#include "src/__support/common.h"
#include "src/threads/call_once.h"
#include "src/threads/linux/Futex.h"

#include <limits.h>
#include <linux/futex.h>
#include <sys/syscall.h> // For syscall numbers.
#include <threads.h>     // For call_once related type definition.

namespace __llvm_libc {

static constexpr FutexWordType START = 0x11;
static constexpr FutexWordType WAITING = 0x22;
static constexpr FutexWordType FINISH = 0x33;
static constexpr once_flag ONCE_FLAG_INIT_VAL = ONCE_FLAG_INIT;

LLVM_LIBC_FUNCTION(void, call_once,
                   (once_flag * flag, __call_once_func_t func)) {
  auto *futex_word = reinterpret_cast<cpp::Atomic<FutexWordType> *>(flag);
  static_assert(sizeof(*futex_word) == sizeof(once_flag));

  FutexWordType not_called = ONCE_FLAG_INIT_VAL.__word;

  // The C standard wording says:
  //
  //     The completion of the function func synchronizes with all
  //     previous or subsequent calls to call_once with the same
  //     flag variable.
  //
  // What this means is that, the call_once call can return only after
  // the called function |func| returns. So, we use futexes to synchronize
  // calls with the same flag value.
  if (futex_word->compare_exchange_strong(not_called, START)) {
    func();
    auto status = futex_word->exchange(FINISH);
    if (status == WAITING) {
      __llvm_libc::syscall_impl(FUTEX_SYSCALL_ID, &futex_word->val,
                                FUTEX_WAKE_PRIVATE,
                                INT_MAX, // Wake all waiters.
                                0, 0, 0);
    }
    return;
  }

  FutexWordType status = START;
  if (futex_word->compare_exchange_strong(status, WAITING) ||
      status == WAITING) {
    __llvm_libc::syscall_impl(
        FUTEX_SYSCALL_ID, &futex_word->val, FUTEX_WAIT_PRIVATE,
        WAITING, // Block only if status is still |WAITING|.
        0, 0, 0);
  }
}

} // namespace __llvm_libc
