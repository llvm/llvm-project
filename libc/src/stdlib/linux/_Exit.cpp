//===------------------- Linux Implementation of _Exit --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/sys/syscall.h"          // For syscall numbers.
#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include "src/stdlib/_Exit.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(void, _Exit, (int status)) {
  for (;;) {
    __llvm_libc::syscall_impl(SYS_exit_group, status);
    __llvm_libc::syscall_impl(SYS_exit, status);
  }
}

} // namespace __llvm_libc
