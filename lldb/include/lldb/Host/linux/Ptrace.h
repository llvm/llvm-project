//===-- Ptrace.h ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// This file defines ptrace functions & structures

#ifndef liblldb_Host_linux_Ptrace_h_
#define liblldb_Host_linux_Ptrace_h_

#include <sys/ptrace.h>

#ifdef __ANDROID_NDK__
#define PT_DETACH PTRACE_DETACH
typedef int __ptrace_request;
#endif

#define DEBUG_PTRACE_MAXBYTES 20

// Support ptrace extensions even when compiled without required kernel support
#ifndef PT_GETREGS
#ifndef PTRACE_GETREGS
#define PTRACE_GETREGS 12
#endif
#endif
#ifndef PT_SETREGS
#ifndef PTRACE_SETREGS
#define PTRACE_SETREGS 13
#endif
#endif
#ifndef PT_GETFPREGS
#ifndef PTRACE_GETFPREGS
#define PTRACE_GETFPREGS 14
#endif
#endif
#ifndef PT_SETFPREGS
#ifndef PTRACE_SETFPREGS
#define PTRACE_SETFPREGS 15
#endif
#endif
#ifndef PTRACE_GETREGSET
#define PTRACE_GETREGSET 0x4204
#endif
#ifndef PTRACE_SETREGSET
#define PTRACE_SETREGSET 0x4205
#endif
#ifndef PTRACE_GET_THREAD_AREA
#define PTRACE_GET_THREAD_AREA 25
#endif
#ifndef PTRACE_ARCH_PRCTL
#define PTRACE_ARCH_PRCTL 30
#endif
#ifndef ARCH_GET_FS
#define ARCH_SET_GS 0x1001
#define ARCH_SET_FS 0x1002
#define ARCH_GET_FS 0x1003
#define ARCH_GET_GS 0x1004
#endif

#define LLDB_PTRACE_NT_ARM_TLS 0x401 // ARM TLS register

#endif // liblldb_Host_linux_Ptrace_h_
