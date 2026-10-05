//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of ppoll.
///
//===----------------------------------------------------------------------===//

#include "src/poll/ppoll.h"
#include "hdr/types/struct_timespec.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/ppoll.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, ppoll,
                   (struct pollfd * fds, nfds_t nfds,
                    const struct timespec *__restrict tmo_p,
                    const sigset_t *__restrict sigmask)) {
  timespec ts;
  timespec *tsp = nullptr;
  if (tmo_p != nullptr) {
    ts = *tmo_p;
    tsp = &ts;
  }
  auto result = linux_syscalls::ppoll(fds, nfds, tsp, sigmask);
  if (!result) {
    libc_errno = result.error();
    return -1;
  }
  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL
