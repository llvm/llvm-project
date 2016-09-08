//===-- HostThreadMacOSX.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Host_macosx_HostThreadMacOSX_h_
#define lldb_Host_macosx_HostThreadMacOSX_h_

#include "lldb/Host/posix/HostThreadPosix.h"

namespace lldb_private {

class HostThreadMacOSX : public HostThreadPosix {
  friend class ThreadLauncher;

public:
  HostThreadMacOSX();
  HostThreadMacOSX(lldb::thread_t thread);

protected:
  static lldb::thread_result_t ThreadCreateTrampoline(lldb::thread_arg_t arg);
};
}

#endif
