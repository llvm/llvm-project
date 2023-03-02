//===-- Linux implementation of syscall -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/syscall.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include <src/errno/libc_errno.h>
#include <stdarg.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(long, __llvm_libc_syscall,
                   (long number, long arg1, long arg2, long arg3, long arg4,
                    long arg5, long arg6)) {
  long ret =
      __llvm_libc::syscall_impl(number, arg1, arg2, arg3, arg4, arg5, arg6);
  // Syscalls may return large positive values that overflow, but will never
  // return values between -4096 and -1
  if (static_cast<unsigned long>(ret) > -4096UL) {
    libc_errno = -ret;
    return -1;
  }
  return ret;
}

} // namespace __llvm_libc
