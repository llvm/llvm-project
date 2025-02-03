//===-- Definition of macros from sys/wait.h ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_MACROS_LINUX_SYS_WAIT_MACROS_H
#define LLVM_LIBC_MACROS_LINUX_SYS_WAIT_MACROS_H

#include <linux/wait.h>

// Wait status info macros
#define __WEXITSTATUS(status) (((status)&0xff00) >> 8)
#define __WTERMSIG(status) ((status)&0x7f)
#define __WIFEXITED(status) (__WTERMSIG(status) == 0)

// Macros for constructing status values.
#define __W_EXITCODE(ret, sig) ((ret) << 8 | (sig))
#define __W_STOPCODE(sig) ((sig) << 8 | 0x7f)
#define __W_CONTINUED 0xffff
#define __WCOREFLAG 0x80

#define WEXITSTATUS(status) __WEXITSTATUS(status)
#define WTERMSIG(status) __WTERMSIG(status)
#define WIFEXITED(status) __WIFEXITED(status)

#define WCOREFLAG __WCOREFLAG
#define W_EXITCODE(ret, sig) __W_EXITCODE(ret, sig)
#define W_STOPCODE(sig) __W_STOPCODE(sig)

#endif // LLVM_LIBC_MACROS_LINUX_SYS_WAIT_MACROS_H
