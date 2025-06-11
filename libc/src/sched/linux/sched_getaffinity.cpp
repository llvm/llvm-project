//===-- Implementation of sched_getaffinity -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sched/sched_getaffinity.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

#include <sched.h>
#include <stdint.h>
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, sched_getaffinity,
                   (pid_t tid, size_t cpuset_size, cpu_set_t *mask)) {
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_sched_getaffinity, tid,
                                              cpuset_size, mask);
  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }
  if (size_t(ret) < cpuset_size) {
    // This means that only |ret| bytes in |mask| have been set. We will have to
    // zero out the remaining bytes.
    auto *mask_bytes = reinterpret_cast<uint8_t *>(mask);
    for (size_t i = size_t(ret); i < cpuset_size; ++i)
      mask_bytes[i] = 0;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
