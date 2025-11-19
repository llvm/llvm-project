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
#include "lldb/Utility/Args.h"
#include "lldb/Utility/Environment.h"
#include "llvm/Support/ErrorOr.h"

namespace lldb_private {

class ProcessLaunchInfo;

class ProcessLauncherWindows : public ProcessLauncher {
public:
  HostProcess LaunchProcess(const ProcessLaunchInfo &launch_info,
                            Status &error) override;

protected:
  HANDLE GetStdioHandle(const ProcessLaunchInfo &launch_info, int fd);

  /// Create a UTF-16 environment block to use with CreateProcessW.
  ///
  /// The buffer is a sequence of null-terminated UTF-16 strings, followed by an
  /// extra L'\0' (two bytes of 0). An empty environment must have one
  /// empty string, followed by an extra L'\0'.
  ///
  /// The keys are sorted to comply with the CreateProcess' calling convention.
  ///
  /// Ensure that the resulting buffer is used in conjunction with
  /// CreateProcessW and be sure that dwCreationFlags includes
  /// CREATE_UNICODE_ENVIRONMENT.
  ///
  /// \param env The Environment object to convert.
  /// \returns The sorted sequence of environment variables and their values,
  /// separated by null terminators.
  static std::vector<wchar_t> CreateEnvironmentBufferW(const Environment &env);

  /// Flattens an Args object into a Windows command-line wide string.
  ///
  /// Returns an empty string if args is empty.
  ///
  /// \param args The Args object to flatten.
  /// \returns A wide string containing the flattened command line.
  static llvm::ErrorOr<std::wstring>
  GetFlattenedWindowsCommandStringW(Args args);
};
}

#endif