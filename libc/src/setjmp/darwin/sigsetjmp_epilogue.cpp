//===-- Implementation of sigsetjmp_epilogue ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/setjmp/sigsetjmp_epilogue.h"
#include "src/__support/OSUtil/syscall.h"
#include "src/__support/common.h"
#include "src/signal/sigprocmask.h"

namespace LIBC_NAMESPACE_DECL {
[[gnu::returns_twice]] int sigsetjmp_epilogue(jmp_buf buffer, int retval) {
  if (retval) {
    // Restore signal mask from the buffer using syscall_impl for macOS
    syscall_impl<long>(SYS_rt_sigprocmask, SIG_SETMASK, &buffer->sigmask, nullptr, sizeof(sigset_t));
  } else {
    // Save the current signal mask to the buffer using syscall_impl for macOS
    syscall_impl<long>(SYS_rt_sigprocmask, SIG_BLOCK, nullptr, &buffer->sigmask, sizeof(sigset_t));
  }
  return retval;
}
} // namespace LIBC_NAMESPACE_DECL
