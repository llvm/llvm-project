//===-- ProcessLauncherPosixFork.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Host_posix_ProcessLauncherPosixFork_h_
#define lldb_Host_posix_ProcessLauncherPosixFork_h_

#include "lldb/Host/ProcessLauncher.h"

namespace lldb_private {

class ProcessLauncherPosixFork : public ProcessLauncher {
public:
  HostProcess LaunchProcess(const ProcessLaunchInfo &launch_info,
                            Status &error) override;
};

} // end of namespace lldb_private

#endif
