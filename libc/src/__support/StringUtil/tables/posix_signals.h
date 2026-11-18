//===-- Map of POSIX signal numbers to strings ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_STRINGUTIL_TABLES_POSIX_SIGNALS_H
#define LLVM_LIBC_SRC___SUPPORT_STRINGUTIL_TABLES_POSIX_SIGNALS_H

#include "hdr/signal_macros.h" // For signal numbers
#include "src/__support/CPP/array.h"
#include "src/__support/StringUtil/message_mapper.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LIBC_INLINE_VAR constexpr MsgTable<22> POSIX_SIGNALS = {
    MsgMapping(SIGHUP, "Hangup"),
    MsgMapping(SIGQUIT, "Quit"),
    MsgMapping(SIGTRAP, "Trace/breakpoint trap"),
    MsgMapping(SIGBUS, "Bus error"),
    MsgMapping(SIGKILL, "Killed"),
    MsgMapping(SIGUSR1, "User defined signal 1"),
    MsgMapping(SIGUSR2, "User defined signal 2"),
    MsgMapping(SIGPIPE, "Broken pipe"),
    MsgMapping(SIGALRM, "Alarm clock"),
    MsgMapping(SIGCHLD, "Child exited"),
    MsgMapping(SIGCONT, "Continued"),
    MsgMapping(SIGSTOP, "Stopped (signal)"),
    MsgMapping(SIGTSTP, "Stopped"),
    MsgMapping(SIGTTIN, "Stopped (tty input)"),
    MsgMapping(SIGTTOU, "Stopped (tty output)"),
    MsgMapping(SIGURG, "Urgent I/O condition"),
    MsgMapping(SIGXCPU, "CPU time limit exceeded"),
    MsgMapping(SIGXFSZ, "File size limit exceeded"),
    MsgMapping(SIGVTALRM, "Virtual timer expired"),
    MsgMapping(SIGPROF, "Profiling timer expired"),
    MsgMapping(SIGPOLL, "I/O possible"),
    MsgMapping(SIGSYS, "Bad system call"),
};

LIBC_INLINE_VAR constexpr MsgTable<22> POSIX_SIGNAL_NAMES = {
    MsgMapping(SIGHUP, "HUP"),       MsgMapping(SIGQUIT, "QUIT"),
    MsgMapping(SIGTRAP, "TRAP"),     MsgMapping(SIGBUS, "BUS"),
    MsgMapping(SIGKILL, "KILL"),     MsgMapping(SIGUSR1, "USR1"),
    MsgMapping(SIGUSR2, "USR2"),     MsgMapping(SIGPIPE, "PIPE"),
    MsgMapping(SIGALRM, "ALRM"),     MsgMapping(SIGCHLD, "CHLD"),
    MsgMapping(SIGCONT, "CONT"),     MsgMapping(SIGSTOP, "STOP"),
    MsgMapping(SIGTSTP, "TSTP"),     MsgMapping(SIGTTIN, "TTIN"),
    MsgMapping(SIGTTOU, "TTOU"),     MsgMapping(SIGURG, "URG"),
    MsgMapping(SIGXCPU, "XCPU"),     MsgMapping(SIGXFSZ, "XFSZ"),
    MsgMapping(SIGVTALRM, "VTALRM"), MsgMapping(SIGPROF, "PROF"),
    MsgMapping(SIGPOLL, "POLL"),     MsgMapping(SIGSYS, "SYS"),
};

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_STRINGUTIL_TABLES_POSIX_SIGNALS_H
