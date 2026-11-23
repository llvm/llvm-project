//===-- Implementation of sched_setaffinity -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sched/sched_setaffinity.h"

#include "src/__support/CPP/span.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/sched_setaffinity.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

#include "hdr/types/cpu_set_t.h"
#include "hdr/types/pid_t.h"
#include "hdr/types/size_t.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, sched_setaffinity,
                   (pid_t tid, size_t cpuset_size, const cpu_set_t *mask)) {
  auto result = linux_syscalls::sched_setaffinity(
      tid, cpp::span<const unsigned char>(
               reinterpret_cast<const unsigned char *>(mask), cpuset_size));
  if (!result) {
    libc_errno = result.error();
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
