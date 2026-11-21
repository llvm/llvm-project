//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Syscall wrapper for lseek.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_LSEEK_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_LSEEK_H

#include "hdr/stdint_proxy.h"
#include "hdr/types/off_t.h"
#include "src/__support/OSUtil/linux/syscall.h" // syscall_checked
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h> // For syscall numbers

namespace LIBC_NAMESPACE_DECL {
namespace linux_syscalls {

LIBC_INLINE ErrorOr<off_t> lseek(int fd, off_t offset, int whence) {
#ifdef SYS_lseek
  LIBC_SANITIZER_SYSCALL_PRE(lseek, fd, offset, whence);
  auto ret = syscall_checked<off_t>(SYS_lseek, fd, offset, whence);
  LIBC_SANITIZER_SYSCALL_POST(lseek, syscall_result_to_long(ret), fd, offset,
                              whence);
  return ret;
#elif defined(SYS_llseek) || defined(SYS__llseek)
#ifdef SYS_llseek
  constexpr long LLSEEK_SYSCALL_NO = SYS_llseek;
#elif defined(SYS__llseek)
  constexpr long LLSEEK_SYSCALL_NO = SYS__llseek;
#endif
  off_t result;
  uint64_t offset_64 = static_cast<uint64_t>(offset);
  long offset_high = static_cast<long>(offset_64 >> 32);
  long offset_low = static_cast<long>(offset_64);
  LIBC_SANITIZER_SYSCALL_PRE(llseek, fd, offset_high, offset_low, &result,
                             whence);
  auto ret = syscall_checked<int>(LLSEEK_SYSCALL_NO, fd, offset_high,
                                  offset_low, &result, whence);
  LIBC_SANITIZER_SYSCALL_POST(llseek, syscall_result_to_long(ret), fd,
                              offset_high, offset_low, &result, whence);
  if (!ret)
    return Error(ret.error());
  return result;
#else
#error "lseek, llseek and _llseek syscalls not available."
#endif
}

} // namespace linux_syscalls
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_SYSCALL_WRAPPERS_LSEEK_H
