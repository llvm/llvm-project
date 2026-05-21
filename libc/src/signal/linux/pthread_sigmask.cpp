//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of pthread_sigmask.
///
//===----------------------------------------------------------------------===//

#include "src/signal/pthread_sigmask.h"

#include "hdr/types/sigset_t.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/rt_sigprocmask.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, pthread_sigmask,
                   (int how, const sigset_t *__restrict set,
                    sigset_t *__restrict oldset)) {
  auto result = linux_syscalls::rt_sigprocmask(how, set, oldset);
  if (result.has_value())
    return 0;

  return result.error();
}

} // namespace LIBC_NAMESPACE_DECL
