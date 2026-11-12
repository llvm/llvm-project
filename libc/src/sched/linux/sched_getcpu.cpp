//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sched/sched_getcpu.h"

#include "src/__support/OSUtil/linux/syscall_wrappers/getcpu.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, sched_getcpu, ()) {
  unsigned int cpu = 0;
  auto result = linux_syscalls::getcpu(&cpu, nullptr);
  if (!result) {
    libc_errno = result.error();
    return -1;
  }
  return cpu;
}

} // namespace LIBC_NAMESPACE_DECL
