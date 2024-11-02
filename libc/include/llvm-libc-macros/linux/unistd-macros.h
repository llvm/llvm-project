//===-- Definition of macros from unistd.h --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_MACROS_LINUX_UNISTD_MACROS_H
#define __LLVM_LIBC_MACROS_LINUX_UNISTD_MACROS_H

// Values for mode argument to the access(...) function.
#define F_OK 0
#define X_OK 1
#define W_OK 2
#define R_OK 4

#define _SC_PAGESIZE 1
#define _SC_PAGE_SIZE _SC_PAGESIZE

// Macro to set up the call to the __llvm_libc_syscall function
// This is to prevent the call from having fewer than 6 arguments, since six
// arguments are always passed to the syscall. Unnecessary arguments are
// ignored.
#define __syscall_helper(sysno, arg1, arg2, arg3, arg4, arg5, arg6, ...)       \
  __llvm_libc_syscall((long)(sysno), (long)(arg1), (long)(arg2), (long)(arg3), \
                      (long)(arg4), (long)(arg5), (long)(arg6))
#define syscall(...) __syscall_helper(__VA_ARGS__, 0, 1, 2, 3, 4, 5, 6)

#endif // __LLVM_LIBC_MACROS_LINUX_UNISTD_MACROS_H
