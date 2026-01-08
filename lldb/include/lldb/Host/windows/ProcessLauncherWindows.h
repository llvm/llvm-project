//===-- ProcessLauncherWindows.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Host_windows_ProcessLauncherWindows_h_
#define lldb_Host_windows_ProcessLauncherWindows_h_

#include "lldb/Host/ProcessLaunchInfo.h"
#include "lldb/Host/ProcessLauncher.h"
#include "lldb/Host/windows/windows.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/WindowsError.h"

#include <optional>

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

  /// Setup the PseudoConsole handle in the underlying
  /// LPPROC_THREAD_ATTRIBUTE_LIST.
  ///
  /// \param hPC
  ///     The handle to the PseudoConsole.
  llvm::Error SetupPseudoConsole(HPCON hPC);

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
  /// \param startupinfoex
  ///   The extended STARTUPINFO structure for the process being created.
  ///
  /// \param launch_info
  ///   The process launch configuration.
  ///
  /// \param stdout_handle
  /// \param stderr_handle
  /// \param stdin_handle
  ///   Optional explicit standard stream handles to use for the child process.
  ///
  /// \returns
  ///   `std::vector<HANDLE>` containing all handles that the child must
  ///   inherit.
  static llvm::ErrorOr<std::vector<HANDLE>>
  GetInheritedHandles(STARTUPINFOEXW &startupinfoex,
                      const ProcessLaunchInfo *launch_info = nullptr,
                      HANDLE stdout_handle = NULL, HANDLE stderr_handle = NULL,
                      HANDLE stdin_handle = NULL);

  static HANDLE GetStdioHandle(const ProcessLaunchInfo &launch_info, int fd);

  /// Creates a file handle suitable for redirecting stdin, stdout,
  /// or stderr of a child process.
  ///
  /// \param path The file path to open. If empty, returns NULL (no
  /// redirection).
  /// \param fd The file descriptor type: STDIN_FILENO, STDOUT_FILENO, or
  /// STDERR_FILENO.
  ///
  /// \return A handle to the opened file, or NULL if the path is empty or the
  /// file
  ///         cannot be opened (INVALID_HANDLE_VALUE is converted to NULL).
  ///
  /// Behavior by file descriptor:
  /// - STDIN_FILENO: Opens existing file for reading (GENERIC_READ,
  /// OPEN_EXISTING).
  /// - STDOUT_FILENO: Creates/truncates file for writing (GENERIC_WRITE,
  /// CREATE_ALWAYS).
  /// - STDERR_FILENO: Creates/truncates file for writing with write-through
  ///                  (FILE_FLAG_WRITE_THROUGH ensures immediate disk writes,
  ///                   bypassing system cache for error messages).
  ///
  /// All handles are created with:
  /// - Inheritance enabled (bInheritHandle = TRUE) so child processes can use
  /// them.
  /// - Shared read/write/delete access to allow other processes to access the
  /// file.
  static HANDLE GetStdioHandle(const llvm::StringRef path, int fd);
};

/// Flattens an Args object into a Windows command-line wide string.
///
/// Returns an empty string if args is empty.
///
/// \param args The Args object to flatten.
/// \returns A wide string containing the flattened command line.
llvm::ErrorOr<std::wstring> GetFlattenedWindowsCommandStringW(Args args);

/// Flattens an Args object into a Windows command-line wide string.
///
/// Returns an empty string if args is empty.
///
/// \param args The Args object to flatten.
/// \returns A wide string containing the flattened command line.
llvm::ErrorOr<std::wstring> GetFlattenedWindowsCommandStringW(char *args[]);

/// Allocate and initialize a PROC_THREAD_ATTRIBUTE_LIST structure
/// that can be used with CreateProcess to specify extended process creation
/// attributes (such as inherited handles).
///
/// \param[in] startupinfoex The STARTUPINFOEXW structure whose lpAttributeList
/// will
///                          be initialized.
///
/// \return On success, returns a scope_exit cleanup object that will
/// automatically
///         delete and free the attribute list when it goes out of scope.
///         On failure, returns the corresponding Windows error code.
llvm::ErrorOr<llvm::scope_exit<std::function<void()>>>
SetupProcThreadAttributeList(STARTUPINFOEXW &startupinfoex);
}

#endif
