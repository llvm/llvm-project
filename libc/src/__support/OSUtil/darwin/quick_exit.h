//===--------- Darwin implementation of a quick exit function ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_OSUTIL_DARWIN_QUICK_EXIT_H
#define LLVM_LIBC_SRC_SUPPORT_OSUTIL_DARWIN_QUICK_EXIT_H

#include "syscall.h" // For internal syscall function.

#include "src/__support/common.h"

namespace __llvm_libc {

LIBC_INLINE void quick_exit(int status) {
  for (;;) {
    __llvm_libc::syscall_impl(1 /* SYS_exit */, status);
  }
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_OSUTIL_DARWIN_QUICK_EXIT_H
