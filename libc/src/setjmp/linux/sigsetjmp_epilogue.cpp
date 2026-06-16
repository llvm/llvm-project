//===-- Implementation of sigsetjmp_epilogue ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/setjmp/sigsetjmp_epilogue.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/rt_sigprocmask.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE_DECL {
[[gnu::returns_twice]] int sigsetjmp_epilogue(sigjmp_buf buffer, int retval) {
  // If set is NULL, then the signal mask is unchanged (i.e., how is
  // ignored), but the current value of the signal mask is nevertheless
  // returned in oldset (if it is not NULL).
  linux_syscalls::rt_sigprocmask(
      SIG_SETMASK,
      /* set= */ retval ? &buffer->sigmask : nullptr,
      /* old_set= */ retval ? nullptr : &buffer->sigmask);
  return retval;
}
} // namespace LIBC_NAMESPACE_DECL
