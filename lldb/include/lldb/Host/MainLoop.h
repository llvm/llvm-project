//===-- MainLoop.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Host_MainLoop_h_
#define lldb_Host_MainLoop_h_

#ifdef _WIN32
#include "lldb/Host/MainLoopBase.h"
namespace lldb_private {
typedef MainLoopBase MainLoop;
}
#else
#include "lldb/Host/posix/MainLoopPosix.h"
namespace lldb_private {
typedef MainLoopPosix MainLoop;
}
#endif

#endif // lldb_Host_MainLoop_h_
