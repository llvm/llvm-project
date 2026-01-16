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

/// This class manages the lifetime of a PROC_THREAD_ATTRIBUTE_LIST, which is
/// used with STARTUPINFOEX.
///
/// The attribute list is automatically cleaned up when this object is
/// destroyed.
class ProcThreadAttributeList {
public:
  /// Allocate memory for the attribute list, initialize it, and sets the
  /// lpAttributeList member of STARTUPINFOEXW structure.
  ///
  /// \param[in,out] startupinfoex
  ///     The STARTUPINFOEXW structure whose lpAttributeList member will be set
  ///     to point to the attribute list. The caller must ensure
  ///     this structure remains valid for the lifetime of the returned object.
  ///
  /// \return
  ///     A ProcThreadAttributeList object on success, or an error code on
  ///     failure.
  static llvm::ErrorOr<ProcThreadAttributeList>
  Create(STARTUPINFOEXW &startupinfoex);

  ~ProcThreadAttributeList() {
    if (lpAttributeList) {
      DeleteProcThreadAttributeList(lpAttributeList);
      free(lpAttributeList);
    }
  }

  /// ProcThreadAttributeList is not copyable.
  /// @{
  ProcThreadAttributeList(const ProcThreadAttributeList &) = delete;
  ProcThreadAttributeList &operator=(const ProcThreadAttributeList &) = delete;
  /// @}

  ProcThreadAttributeList(ProcThreadAttributeList &&other) noexcept
      : lpAttributeList(other.lpAttributeList) {
    other.lpAttributeList = nullptr;
  }

private:
  explicit ProcThreadAttributeList(LPPROC_THREAD_ATTRIBUTE_LIST list)
      : lpAttributeList(list) {}

  LPPROC_THREAD_ATTRIBUTE_LIST lpAttributeList;
};

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
