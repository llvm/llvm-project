//===-- Ptrace.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This file defines ptrace functions & structures

#ifndef LIBLLDB_HOST_AIX_PTRACE_H_
#define LIBLLDB_HOST_AIX_PTRACE_H_

#include <sys/ptrace.h>

// Support ptrace extensions even when compiled without required kernel support
#ifndef PTRACE_GETREGS
#define PTRACE_GETREGS (PT_COMMAND_MAX + 1)
#endif
#ifndef PTRACE_SETREGS
#define PTRACE_SETREGS (PT_COMMAND_MAX + 2)
#endif
#ifndef PTRACE_GETFPREGS
#define PTRACE_GETFPREGS (PT_COMMAND_MAX + 3)
#endif
#ifndef PTRACE_SETFPREGS
#define PTRACE_SETFPREGS (PT_COMMAND_MAX + 4)
#endif
#ifndef PTRACE_GETREGSET
#define PTRACE_GETREGSET 0x4204
#endif
#ifndef PTRACE_SETREGSET
#define PTRACE_SETREGSET 0x4205
#endif
#ifndef PTRACE_GETVRREGS
#define PTRACE_GETVRREGS (PT_COMMAND_MAX + 5)
#endif
#ifndef PTRACE_GETVSRREGS
#define PTRACE_GETVSRREGS (PT_COMMAND_MAX + 6)
#endif

#endif // LIBLLDB_HOST_AIX_PTRACE_H_
