//===------------- Darwin implementation of IO utils ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_OSUTIL_DARWIN_IO_H
#define LLVM_LIBC_SRC_SUPPORT_OSUTIL_DARWIN_IO_H

#include "src/__support/CPP/string_view.h"
#include "syscall.h" // For internal syscall function.

namespace __llvm_libc {

LIBC_INLINE void write_to_stderr(cpp::string_view msg) {
  __llvm_libc::syscall_impl(4 /*SYS_write*/, 2 /* stderr */,
                            reinterpret_cast<long>(msg.data()), msg.size());
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_OSUTIL_DARWIN_IO_H
