//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Syscall wrapper for getegid.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_GETEGID_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_GETEGID_H

#include "hdr/types/gid_t.h"
#include "src/__support/OSUtil/linux/syscall.h" // syscall_impl
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

LIBC_INLINE gid_t getegid() {
  // getegid is expected to always succeed, thus no error checking here.
#if defined(SYS_getegid32)
  LIBC_SANITIZER_SYSCALL_PRE_0(getegid32);
  auto syscall_ret = syscall_impl<gid_t>(SYS_getegid32);
  LIBC_SANITIZER_SYSCALL_POST(getegid32, static_cast<long>(syscall_ret));
  return syscall_ret;
#elif defined(SYS_getegid)
  LIBC_SANITIZER_SYSCALL_PRE_0(getegid);
  auto syscall_ret = syscall_impl<gid_t>(SYS_getegid);
  LIBC_SANITIZER_SYSCALL_POST(getegid, static_cast<long>(syscall_ret));
  return syscall_ret;
#else
#error "SYS_getegid and SYS_getegid32 syscalls not available."
#endif
}

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_GETEGID_H
