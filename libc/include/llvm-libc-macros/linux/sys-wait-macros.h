//===-- Definition of macros from sys/wait.h ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_MACROS_LINUX_SYS_WAIT_MACROS_H
#define __LLVM_LIBC_MACROS_LINUX_SYS_WAIT_MACROS_H

// Wait flags
#define WNOHANG 1    // Do not block
#define WUNTRACED 2  // Report is a child has stopped even if untraced
#define WEXITED 4    // Report dead child
#define WCONTINUED 8 // Report if a stopped child has been resumed by SIGCONT
#define WSTOPPED WUNTRACED

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

// First argument to waitid:
#define P_ALL 0
#define P_PID 1
#define P_PGID 2
#define P_PIDFD 3

#endif // __LLVM_LIBC_MACROS_LINUX_SYS_WAIT_MACROS_H
