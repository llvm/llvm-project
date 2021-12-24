//===-------------- Linux implementation of IO utils ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_OSUTIL_LINUX_IO_H
#define LLVM_LIBC_SRC_SUPPORT_OSUTIL_LINUX_IO_H

#include "include/sys/syscall.h" // For syscall numbers.
#include "src/string/string_utils.h"
#include "syscall.h" // For internal syscall function.

namespace __llvm_libc {

static inline void write_to_stderr(const char *msg) {
  __llvm_libc::syscall(SYS_write, 2 /* stderr */, msg,
                       internal::string_length(msg));
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_OSUTIL_LINUX_IO_H
