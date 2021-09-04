//===-- Linux implementation of the call_once function --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/threads/call_once.h"
#include "config/linux/syscall.h" // For syscall functions.
#include "include/sys/syscall.h"  // For syscall numbers.
#include "include/threads.h"      // For call_once related type definition.
#include "src/__support/common.h"
#include "src/threads/linux/Futex.h"

#include <limits.h>
#include <linux/futex.h>
#include <stdatomic.h>

namespace __llvm_libc {

static constexpr unsigned START = 0x11;
static constexpr unsigned WAITING = 0x22;
static constexpr unsigned FINISH = 0x33;

LLVM_LIBC_FUNCTION(void, call_once,
                   (once_flag * flag, __call_once_func_t func)) {
  FutexWord *futex_word = reinterpret_cast<FutexWord *>(flag);
  unsigned int not_called = ONCE_FLAG_INIT;

  // The C standard wording says:
  //
  //     The completion of the function func synchronizes with all
  //     previous or subsequent calls to call_once with the same
  //     flag variable.
  //
  // What this means is that, the call_once call can return only after
  // the called function |func| returns. So, we use futexes to synchronize
  // calls with the same flag value.
  if (::atomic_compare_exchange_strong(futex_word, &not_called, START)) {
    func();
    auto status = ::atomic_exchange(futex_word, FINISH);
    if (status == WAITING) {
      __llvm_libc::syscall(SYS_futex, futex_word, FUTEX_WAKE_PRIVATE,
                           INT_MAX, // Wake all waiters.
                           0, 0, 0);
    }
    return;
  }

  unsigned int status = START;
  if (::atomic_compare_exchange_strong(futex_word, &status, WAITING) ||
      status == WAITING) {
    __llvm_libc::syscall(SYS_futex, futex_word, FUTEX_WAIT_PRIVATE,
                         WAITING, // Block only if status is still |WAITING|.
                         0, 0, 0);
  }
}

} // namespace __llvm_libc
