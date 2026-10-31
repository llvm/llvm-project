//===-- Linux implementation of sigpending -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/signal/sigpending.h"

#include "hdr/types/sigset_t.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/rt_sigpending.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, sigpending, (sigset_t *set)) {
  auto result = linux_syscalls::rt_sigpending(set);
  if (result.has_value())
    return 0;

  libc_errno = result.error();
  return -1;
}

} // namespace LIBC_NAMESPACE_DECL