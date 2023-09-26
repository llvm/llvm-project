//===---------- Linux implementation of a quick exit function ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_QUICK_EXIT_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_QUICK_EXIT_H

#include "syscall.h"             // For internal syscall function.

#include "src/__support/common.h"

#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE {

LIBC_INLINE void quick_exit(int status) {
  for (;;) {
    LIBC_NAMESPACE::syscall_impl<long>(SYS_exit_group, status);
    LIBC_NAMESPACE::syscall_impl<long>(SYS_exit, status);
  }
}

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_QUICK_EXIT_H
