//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of sigwaitinfo.
///
//===----------------------------------------------------------------------===//

#include "src/signal/sigwaitinfo.h"

#include "hdr/types/siginfo_t.h"
#include "hdr/types/sigset_t.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/rt_sigtimedwait.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, sigwaitinfo,
                   (const sigset_t *__restrict set,
                    siginfo_t *__restrict info)) {
  // Linux kernel specifically design rt_sigtimedwait syscall with a nullptr
  // as sigwaitinfo behaviour, even if it was left unspecified by POSIX norm.
  // See in the notes section of:
  // https://man7.org/linux/man-pages/man2/rt_sigtimedwait.2.html
  auto result = linux_syscalls::rt_sigtimedwait(set, info, nullptr);
  if (!result.has_value()) {
    libc_errno = result.error();
    return -1;
  }
  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL
