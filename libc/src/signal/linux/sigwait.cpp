//===-- Linux implementation of sigwait -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/signal/sigwait.h"

#include "hdr/errno_macros.h"
#include "hdr/types/siginfo_t.h"
#include "hdr/types/sigset_t.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/rt_sigtimedwait.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, sigwait,
                   (const sigset_t *__restrict set, int *__restrict sig)) {
  siginfo_t info;
  ErrorOr<int> result = 0;
  do {
    result = linux_syscalls::rt_sigtimedwait(set, &info, nullptr);
  } while (!result.has_value() && result.error() == EINTR);

  if (!result.has_value())
    return result.error();

  *sig = info.si_signo;
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL