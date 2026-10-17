//===-- Darwin implementation of sigprocmask ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/signal/sigprocmask.h"
#include "src/__support/OSUtil/darwin/syscall_wrappers/sigprocmask.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, sigprocmask,
                   (int how, const sigset_t *__restrict set,
                    sigset_t *__restrict oldset)) {
  auto result = darwin_syscalls::sigprocmask(how, set, oldset);
  if (result.has_value())
    return result.value();

  libc_errno = result.error();
  return -1;
}

} // namespace LIBC_NAMESPACE_DECL
