//===-- Implementation of sched_getaffinity -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sched/sched_getaffinity.h"

#include "hdr/stdint_proxy.h"
#include "src/__support/CPP/span.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/sched_getaffinity.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

#include "hdr/types/cpu_set_t.h"
#include "hdr/types/pid_t.h"
#include "hdr/types/size_t.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, sched_getaffinity,
                   (pid_t tid, size_t cpuset_size, cpu_set_t *mask)) {
  auto result = linux_syscalls::sched_getaffinity(
      tid, cpp::span<unsigned char>(reinterpret_cast<unsigned char *>(mask),
                                    cpuset_size));
  if (!result) {
    libc_errno = result.error();
    return -1;
  }
  int ret = result.value();
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
