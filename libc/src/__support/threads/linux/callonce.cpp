//===-- Linux implementation of the callonce function ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/threads/callonce.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/linux/callonce.h"
#include "src/__support/threads/linux/futex_utils.h"

namespace LIBC_NAMESPACE_DECL {
namespace callonce_impl {
int callonce_slowpath(CallOnceFlag *flag, CallOnceCallback *func) {
  auto *futex_word = reinterpret_cast<Futex *>(flag);

  FutexWordType not_called = NOT_CALLED;

  // The call_once call can return only after the called function |func|
  // returns. So, we use futexes to synchronize calls with the same flag value.
  if (futex_word->compare_exchange_strong(not_called, START)) {
    func();
    auto status = futex_word->exchange(FINISH);
    if (status == WAITING)
      futex_word->notify_all();
    return 0;
  }

  FutexWordType status = START;
  if (futex_word->compare_exchange_strong(status, WAITING) ||
      status == WAITING) {
    futex_word->wait(WAITING);
  }

  return 0;
}
} // namespace callonce_impl
} // namespace LIBC_NAMESPACE_DECL
