//===-- Config.h ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_CONFIG_H
#define LLDB_HOST_CONFIG_H
 
#if defined(__APPLE__)

// This block of code only exists to keep the Xcode project working in the
// absence of a configuration step.
#define LLDB_CONFIG_TERMIOS_SUPPORTED 1

#define HAVE_SYS_EVENT_H 1

#else

#error This file is only used by the Xcode build.

#endif

#endif // #ifndef LLDB_HOST_CONFIG_H
