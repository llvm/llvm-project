//===-- Internal header for Linux abort -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_LINUX_ABORT_UTILS_H
#define LLVM_LIBC_SRC_STDLIB_LINUX_ABORT_UTILS_H

#include "hdr/types/sigset_t.h"
#include "hdr/types/struct_sigaction.h"
#include "src/__support/OSUtil/exit.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/raise.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/signal/linux/signal_utils.h"

namespace LIBC_NAMESPACE_DECL {

namespace abort_utils {
[[noreturn]] LIBC_INLINE void abort() {
  // Try to raise SIGABRT.
  // If this fails, or if a handler returns, keep going with the hard-abort
  // sequence below.
  linux_syscalls::raise(SIGABRT);

  // We get back from abort, potentially from a abort handler.
  // We recover the handler to default and raise it again. Since this is the
  // real abort routine, we demand exclusive access to the abort lock.
  // We have already returned from the first raise, so it is okay to grab
  // exclusive access.
  SigAbortGuard guard(true);
  struct sigaction sa{};
  sa.sa_handler = SIG_DFL;
  sa.sa_flags = 0;
  // There is no recovery path from sigaction failure while aborting.
  unchecked_sigaction(SIGABRT, &sa, nullptr);
  // If this still returns, fall through to the final termination path.
  linux_syscalls::raise(SIGABRT);

  // Now unblock the signal. The pending abort signal is now unblocked and
  // should be delivered to its default handler.
  // If this fails, there is still no meaningful recovery path while aborting.
  unblock_signal(SIGABRT);

  internal::exit(127);
}
} // namespace abort_utils

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDLIB_LINUX_ABORT_UTILS_H
