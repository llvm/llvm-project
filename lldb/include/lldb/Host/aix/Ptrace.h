//===-- Ptrace.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This file defines ptrace functions & structures

#ifndef liblldb_Host_aix_Ptrace_h_
#define liblldb_Host_aix_Ptrace_h_

#include <sys/ptrace.h>

#define DEBUG_PTRACE_MAXBYTES 20

// Support ptrace extensions even when compiled without required kernel support
#ifndef PTRACE_GETREGS
#define PTRACE_GETREGS          (PT_COMMAND_MAX+1)
#endif
#ifndef PTRACE_SETREGS
#define PTRACE_SETREGS          (PT_COMMAND_MAX+2)
#endif
#ifndef PTRACE_GETFPREGS
#define PTRACE_GETFPREGS        (PT_COMMAND_MAX+3)
#endif
#ifndef PTRACE_SETFPREGS
#define PTRACE_SETFPREGS        (PT_COMMAND_MAX+4)
#endif
#ifndef PTRACE_GETREGSET
#define PTRACE_GETREGSET 0x4204
#endif
#ifndef PTRACE_SETREGSET
#define PTRACE_SETREGSET 0x4205
#endif
#ifndef PTRACE_GET_THREAD_AREA
#define PTRACE_GET_THREAD_AREA  (PT_COMMAND_MAX+5)
#endif
#ifndef PTRACE_ARCH_PRCTL
#define PTRACE_ARCH_PRCTL       (PT_COMMAND_MAX+6)
#endif
#ifndef ARCH_GET_FS
#define ARCH_SET_GS 0x1001
#define ARCH_SET_FS 0x1002
#define ARCH_GET_FS 0x1003
#define ARCH_GET_GS 0x1004
#endif
#ifndef PTRACE_PEEKMTETAGS
#define PTRACE_PEEKMTETAGS      (PT_COMMAND_MAX+7)
#endif
#ifndef PTRACE_POKEMTETAGS
#define PTRACE_POKEMTETAGS      (PT_COMMAND_MAX+8)
#endif
#ifndef PTRACE_GETVRREGS
#define PTRACE_GETVRREGS        (PT_COMMAND_MAX+9)
#endif
#ifndef PTRACE_GETVSRREGS
#define PTRACE_GETVSRREGS       (PT_COMMAND_MAX+10)
#endif

#endif // liblldb_Host_aix_Ptrace_h_
