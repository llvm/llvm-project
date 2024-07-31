//===-- Linux implementation of pread -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/pread.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/sanitizer.h" // for MSAN_UNPOISON
#include "src/errno/libc_errno.h"
#include <stdint.h>      // For uint64_t.
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(ssize_t, pread,
                   (int fd, void *buf, size_t count, off_t offset)) {
  ssize_t ret;
  if constexpr (sizeof(long) == sizeof(uint32_t) &&
                sizeof(off_t) == sizeof(uint64_t)) {
    // This is a 32-bit system with a 64-bit offset, offset must be split.
    const uint64_t bits = cpp::bit_cast<uint64_t>(offset);
    const uint32_t lo = bits & UINT32_MAX;
    const uint32_t hi = bits >> 32;
    const long offset_low = cpp::bit_cast<long>(static_cast<long>(lo));
    const long offset_high = cpp::bit_cast<long>(static_cast<long>(hi));
    ret = LIBC_NAMESPACE::syscall_impl<ssize_t>(SYS_pread64, fd, buf, count,
                                                offset_low, offset_high);
  } else {
    ret = LIBC_NAMESPACE::syscall_impl<ssize_t>(SYS_pread64, fd, buf, count,
                                                offset);
  }
  // The cast is important since there is a check that dereferences the pointer
  // which fails on void*.
  MSAN_UNPOISON(reinterpret_cast<char *>(buf), count);
  if (ret < 0) {
    libc_errno = static_cast<int>(-ret);
    return -1;
  }
  return ret;
}

} // namespace LIBC_NAMESPACE_DECL
