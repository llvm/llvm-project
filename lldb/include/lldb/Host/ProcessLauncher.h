//===-- ProcessLauncher.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Host_ProcessLauncher_h_
#define lldb_Host_ProcessLauncher_h_

namespace lldb_private {

class ProcessLaunchInfo;
class Status;
class HostProcess;

class ProcessLauncher {
public:
  virtual ~ProcessLauncher() {}
  virtual HostProcess LaunchProcess(const ProcessLaunchInfo &launch_info,
                                    Status &error) = 0;
};
}

#endif
