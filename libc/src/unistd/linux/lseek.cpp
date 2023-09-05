//===-- Linux implementation of lseek -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/lseek.h"
#include "src/errno/libc_errno.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include <stdint.h>
#include <sys/syscall.h> // For syscall numbers.
#include <unistd.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(off_t, lseek, (int fd, off_t offset, int whence)) {
  off_t result;
#ifdef SYS_lseek
  int ret = __llvm_libc::syscall_impl<int>(SYS_lseek, fd, offset, whence);
  result = ret;
#elif defined(SYS_llseek) || defined(SYS__llseek)
#ifdef SYS_llseek
  constexpr long LLSEEK_SYSCALL_NO = SYS_llseek;
#elif defined(SYS__llseek)
  constexpr long LLSEEK_SYSCALL_NO = SYS__llseek;
#endif
  uint64_t offset_64 = static_cast<uint64_t>(offset);
  int ret = __llvm_libc::syscall_impl<int>(
      LLSEEK_SYSCALL_NO, fd, offset_64 >> 32, offset_64, &result, whence);
#else
#error "lseek, llseek and _llseek syscalls not available."
#endif

  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }
  return result;
}

} // namespace __llvm_libc
