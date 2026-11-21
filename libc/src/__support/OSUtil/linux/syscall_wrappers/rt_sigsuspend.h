//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Syscall wrapper for rt_sigsuspend.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_RT_SIGSUSPEND_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_RT_SIGSUSPEND_H

#include "hdr/types/sigset_t.h"
#include "src/__support/OSUtil/linux/syscall.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h>

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

LIBC_INLINE ErrorOr<int> rt_sigsuspend(const sigset_t *set) {
  LIBC_SANITIZER_SYSCALL_PRE(rt_sigsuspend, set, sizeof(sigset_t));
  auto syscall_ret =
      syscall_checked<int>(SYS_rt_sigsuspend, set, sizeof(sigset_t));
  LIBC_SANITIZER_SYSCALL_POST(rt_sigsuspend,
                              syscall_result_to_long(syscall_ret), set,
                              sizeof(sigset_t));
  return syscall_ret;
}

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_RT_SIGSUSPEND_H
