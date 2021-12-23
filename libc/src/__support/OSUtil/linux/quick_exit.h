//===---------- Linux implementation of a quick exit function ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_OSUTIL_LINUX_QUICK_EXIT_H
#define LLVM_LIBC_SRC_SUPPORT_OSUTIL_LINUX_QUICK_EXIT_H

#include "include/sys/syscall.h" // For syscall numbers.
#include "syscall.h"             // For internal syscall function.

namespace __llvm_libc {

static inline void quick_exit(int status) {
  for (;;) {
    __llvm_libc::syscall(SYS_exit_group, status);
    __llvm_libc::syscall(SYS_exit, status);
  }
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_OSUTIL_LINUX_QUICK_EXIT_H
