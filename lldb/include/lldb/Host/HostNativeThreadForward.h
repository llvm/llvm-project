//===-- HostNativeThreadForward.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Host_HostNativeThreadForward_h_
#define lldb_Host_HostNativeThreadForward_h_

namespace lldb_private {
#if defined(_WIN32)
class HostThreadWindows;
typedef HostThreadWindows HostNativeThread;
#elif defined(__linux__)
class HostThreadLinux;
typedef HostThreadLinux HostNativeThread;
#elif defined(__FreeBSD__) || defined(__FreeBSD_kernel__)
class HostThreadFreeBSD;
typedef HostThreadFreeBSD HostNativeThread;
#elif defined(__NetBSD__)
class HostThreadNetBSD;
typedef HostThreadNetBSD HostNativeThread;
#elif defined(__APPLE__)
class HostThreadMacOSX;
typedef HostThreadMacOSX HostNativeThread;
#endif
}

#endif
