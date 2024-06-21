//===------------ Linux implementation of an exit function ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/common.h"
#include "syscall.h"     // For internal syscall function.
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE::internal {

// mark as no_stack_protector for x86 since TLS can be torn down before calling
// exit so that the stack protector canary cannot be loaded.
#ifdef LIBC_TARGET_ARCH_IS_X86
__attribute__((no_stack_protector))
#endif
__attribute__((noreturn)) void
exit(int status) {
  for (;;) {
    LIBC_NAMESPACE::syscall_impl<long>(SYS_exit_group, status);
    LIBC_NAMESPACE::syscall_impl<long>(SYS_exit, status);
  }
}

} // namespace LIBC_NAMESPACE::internal
