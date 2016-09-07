//===-- ProcessLauncherWindows.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Host_windows_ProcessLauncherWindows_h_
#define lldb_Host_windows_ProcessLauncherWindows_h_

#include "lldb/Host/ProcessLauncher.h"
#include "lldb/Host/windows/windows.h"

namespace lldb_private {

class ProcessLaunchInfo;

class ProcessLauncherWindows : public ProcessLauncher {
public:
  virtual HostProcess LaunchProcess(const ProcessLaunchInfo &launch_info,
                                    Error &error);

protected:
  HANDLE GetStdioHandle(const ProcessLaunchInfo &launch_info, int fd);
};
}

#endif
