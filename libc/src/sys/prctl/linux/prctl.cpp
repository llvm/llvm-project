//===---------- Linux implementation of the prctl function ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/prctl/prctl.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.

#include "src/errno/libc_errno.h"
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, prctl,
                   (int option, unsigned long arg2, unsigned long arg3,
                    unsigned long arg4, unsigned long arg5)) {
  long ret =
      LIBC_NAMESPACE::syscall_impl(SYS_prctl, option, arg2, arg3, arg4, arg5);
  if (ret < 0) {
    libc_errno = static_cast<int>(-ret);
    return -1;
  }
  return static_cast<int>(ret);
}

} // namespace LIBC_NAMESPACE
