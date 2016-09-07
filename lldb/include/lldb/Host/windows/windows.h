//===-- lldb-windows.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_lldb_windows_h_
#define LLDB_lldb_windows_h_

#define NTDDI_VERSION NTDDI_VISTA
#define _WIN32_WINNT _WIN32_WINNT_VISTA
#define WIN32_LEAN_AND_MEAN
#define NOGDI
#define NOMINMAX
#include <windows.h>
#undef GetUserName
#undef LoadImage
#undef CreateProcess
#undef far
#undef near
#undef FAR
#undef NEAR
#define FAR
#define NEAR

#endif // LLDB_lldb_windows_h_
