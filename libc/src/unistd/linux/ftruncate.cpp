//===-- Linux implementation of ftruncate ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/ftruncate.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include "hdr/unistd_macros.h"
#include "src/__support/macros/config.h"
#include "src/errno/libc_errno.h"
#include <stdint.h>      // For uint64_t.
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, ftruncate, (int fd, off_t len)) {
#ifdef SYS_ftruncate
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_ftruncate, fd, len);
#elif defined(SYS_ftruncate64)
  // Same as ftruncate but can handle large offsets
  static_assert(sizeof(off_t) == 8);
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_ftruncate64, fd, (long)len,
                                              (long)(((uint64_t)(len)) >> 32));
#else
#error "ftruncate and ftruncate64 syscalls not available."
#endif

  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
