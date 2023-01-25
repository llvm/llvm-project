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
#define WCONTINUED 8 // Report if a stopped child has been resumed by SIGCONT

// Wait status info macros
#define WTERMSIG(status) (((status)&0x7F))
#define WIFEXITED(status) (WTERMSIG(status) == 0)
#define WEXITSTATUS(status) (((status)&0xFF00) >> 8)
#define WIFSIGNALED(status)                                                    \
  ((WTERMSIG(status) < 0x7F) && (WTERMSIG(status) > 0))

#endif // __LLVM_LIBC_MACROS_LINUX_SYS_WAIT_MACROS_H
