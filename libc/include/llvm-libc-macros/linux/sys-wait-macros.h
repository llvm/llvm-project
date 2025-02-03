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
#define __WEXITSTATUS(status) (((status) & 0xff00) >> 8)
#define __WTERMSIG(status) ((status) & 0x7f)
#define __WIFEXITED(status) (__WTERMSIG(status) == 0)
#define __WIFSIGNALED(status) ((__WTERMSIG(status) + 1) >= 2)
#define __WIFSTOPPED(status) (__WTERMSIG(status) == 0x7f)
#define __WIFCONTINUED(status) ((status) == __W_CONTINUED)
#define __WCOREDUMP(status) ((status) & __WCOREFLAG)

// Macros for constructing status values.
#define __W_EXITCODE(ret, sig) ((ret) << 8 | (sig))
#define __W_STOPCODE(sig) ((sig) << 8 | 0x7f)
#define __W_CONTINUED 0xffff
#define __WCOREFLAG 0x80

#define WCOREDUMP(status) ((status) & __WCOREFLAG)
#define WEXITSTATUS(status) __WEXITSTATUS(status)
#define WIFCONTINUED(status) __WIFCONTINUED(status)
#define WIFEXITED(status) __WIFEXITED(status)
#define WIFSIGNALED(status) __WIFSIGNALED(status)
#define WIFSTOPPED(status) __WIFSTOPPED(status)
#define WSTOPSIG(status) WEXITSTATUS(status)
#define WTERMSIG(status) __WTERMSIG(status)

#define WCOREFLAG __WCOREFLAG
#define W_EXITCODE(ret, sig) __W_EXITCODE(ret, sig)
#define W_STOPCODE(sig) __W_STOPCODE(sig)

#endif // LLVM_LIBC_MACROS_LINUX_SYS_WAIT_MACROS_H
