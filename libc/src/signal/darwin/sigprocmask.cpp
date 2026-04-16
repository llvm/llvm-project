//===-- Darwin implementation of sigprocmask ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/signal/sigprocmask.h"

#include "hdr/stdint_proxy.h"
#include "hdr/types/sigset_t.h"
#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, sigprocmask,
                   (int how, const sigset_t *__restrict set,
                    sigset_t *__restrict oldset)) {
  uint32_t kernel_set = 0;
  uint32_t kernel_oldset = 0;

  if (set)
    kernel_set = static_cast<uint32_t>(set->__signals[0] & 0x7FFFFFFFU);

  int ret = LIBC_NAMESPACE::syscall_impl<int>(
      SYS_sigprocmask, how, set ? &kernel_set : nullptr,
      oldset ? &kernel_oldset : nullptr);
  if (ret == 0) {
    if (oldset)
      oldset->__signals[0] =
          static_cast<unsigned long>(kernel_oldset & 0x7FFFFFFFU);
    return 0;
  }

  libc_errno = ret;
  return -1;
}

} // namespace LIBC_NAMESPACE_DECL
