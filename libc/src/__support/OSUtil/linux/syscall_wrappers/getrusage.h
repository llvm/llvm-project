//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for getrusage
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_SYSCALL_WRAPPERS_GETRUSAGE_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_SYSCALL_WRAPPERS_GETRUSAGE_H

#include "hdr/types/struct_rusage.h"
#include "src/__support/OSUtil/linux/syscall.h" // For syscall_checked
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

LIBC_INLINE ErrorOr<long> getrusage(int who, struct rusage *ru) {
  return syscall_checked<long>(SYS_getrusage, who, ru);
}

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL

#endif  // LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_SYSCALL_WRAPPERS_GETRUSAGE_H
