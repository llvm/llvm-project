//===-- Config.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_CONFIG_H
#define LLDB_HOST_CONFIG_H
 
#if defined(__APPLE__)

// This block of code only exists to keep the Xcode project working in the
// absence of a configuration step.
#define LLDB_CONFIG_TERMIOS_SUPPORTED 1

#define LLDB_EDITLINE_USE_WCHAR 1

#define LLDB_HAVE_EL_RFUNC_T 1

#define HAVE_SYS_EVENT_H 1

#define HAVE_PPOLL 0

#define HAVE_SIGACTION 1

#define HAVE_LIBCOMPRESSION 1

#else

#error This file is only used by the Xcode build.

#endif

#endif // #ifndef LLDB_HOST_CONFIG_H
