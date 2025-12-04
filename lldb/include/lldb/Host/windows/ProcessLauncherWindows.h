//===-- ProcessLauncherWindows.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Host_windows_ProcessLauncherWindows_h_
#define lldb_Host_windows_ProcessLauncherWindows_h_

#include "lldb/Host/ProcessLauncher.h"
#include "lldb/Host/windows/windows.h"
#include "llvm/Support/ErrorOr.h"

namespace lldb_private {

class ProcessLaunchInfo;

class ProcessLauncherWindows : public ProcessLauncher {
public:
  HostProcess LaunchProcess(const ProcessLaunchInfo &launch_info,
                            Status &error) override;

protected:
  HANDLE GetStdioHandle(const ProcessLaunchInfo &launch_info, int fd);

  /// Get the list of Windows handles that should be inherited by the child
  /// process and update `STARTUPINFOEXW` with the handle list.
  ///
  /// If no handles need to be inherited, an empty vector is returned.
  ///
  /// Otherwise, the function populates the
  /// `PROC_THREAD_ATTRIBUTE_HANDLE_LIST` attribute in `startupinfoex` with the
  /// collected handles using `UpdateProcThreadAttribute`. On success, the
  /// vector of inherited handles is returned.
  ///
  /// \param launch_info
  ///   The process launch configuration.
  ///
  /// \param startupinfoex
  ///   The extended STARTUPINFO structure for the process being created.
  ///
  /// \param stdout_handle
  /// \param stderr_handle
  /// \param stdin_handle
  ///   Optional explicit standard stream handles to use for the child process.
  ///
  /// \returns
  ///   `std::vector<HANDLE>` containing all handles that the child must
  ///   inherit.
  llvm::ErrorOr<std::vector<HANDLE>>
  GetInheritedHandles(const ProcessLaunchInfo &launch_info,
                      STARTUPINFOEXW &startupinfoex,
                      HANDLE stdout_handle = NULL, HANDLE stderr_handle = NULL,
                      HANDLE stdin_handle = NULL);
};
}

#endif
