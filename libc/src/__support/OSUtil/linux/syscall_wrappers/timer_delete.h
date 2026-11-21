//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for timer_delete syscall wrapper.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_TIMER_DELETE_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_TIMER_DELETE_H

#include "hdr/types/timer_t.h"
#include "src/__support/OSUtil/linux/syscall.h" // For syscall_checked
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

LIBC_INLINE ErrorOr<int> timer_delete(timer_t timerid) {
#ifdef SYS_timer_delete
  LIBC_SANITIZER_SYSCALL_PRE(timer_delete, timerid);
  auto syscall_ret = syscall_checked<int>(SYS_timer_delete, timerid);
  LIBC_SANITIZER_SYSCALL_POST(timer_delete, syscall_result_to_long(syscall_ret),
                              timerid);
  return syscall_ret;
#else
#error "SYS_timer_delete syscall not available."
#endif
}

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_TIMER_DELETE_H
