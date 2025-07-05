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
  syscall_impl<long>(sigprocmask, SIG_SETMASK,
                     /* set= */ retval ? &buffer->sigmask : nullptr,
                     /* old_set= */ retval ? nullptr : &buffer->sigmask);
  return retval;
}
} // namespace LIBC_NAMESPACE_DECL
