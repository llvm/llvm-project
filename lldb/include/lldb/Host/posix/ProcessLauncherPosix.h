//===-- ProcessLauncherPosix.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Host_posix_ProcessLauncherPosix_h_
#define lldb_Host_posix_ProcessLauncherPosix_h_

#include "lldb/Host/ProcessLauncher.h"

namespace lldb_private {

class ProcessLauncherPosix : public ProcessLauncher {
public:
  HostProcess LaunchProcess(const ProcessLaunchInfo &launch_info,
                            Error &error) override;
};
}

#endif
