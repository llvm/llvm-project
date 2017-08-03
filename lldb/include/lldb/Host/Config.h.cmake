//===-- Config.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_CONFIG_H
#define LLDB_HOST_CONFIG_H

#cmakedefine LLDB_CONFIG_TERMIOS_SUPPORTED

#cmakedefine LLDB_DISABLE_POSIX

#cmakedefine01 HAVE_SYS_EVENT_H

#cmakedefine01 HAVE_PPOLL

#cmakedefine01 HAVE_SIGACTION

#cmakedefine01 HAVE_PROCESS_VM_READV

#cmakedefine01 HAVE_NR_PROCESS_VM_READV

#endif // #ifndef LLDB_HOST_CONFIG_H
