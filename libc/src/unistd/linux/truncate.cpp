//===-- Linux implementation of truncate ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/truncate.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/errno/libc_errno.h"

#include <stdint.h>      // For uint64_t.
#include <sys/syscall.h> // For syscall numbers.
#include <unistd.h>

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, truncate, (const char *path, off_t len)) {
#ifdef SYS_truncate
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_truncate, path, len);
#elif defined(SYS_truncate64)
  // Same as truncate but can handle large offsets
  static_assert(sizeof(off_t) == 8);
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_truncate64, path, (long)len,
                                              (long)(((uint64_t)(len)) >> 32));
#else
#error "truncate and truncate64 syscalls not available."
#endif
  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE
