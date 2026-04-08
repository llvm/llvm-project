//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_WINDOWS_WINDOWSFILEACTION_H
#define LLDB_HOST_WINDOWS_WINDOWSFILEACTION_H

#include "lldb/Host/FileAction.h"
#include "lldb/lldb-types.h"

typedef void *HANDLE;

namespace lldb_private {

/// A Windows-specific extension of FileAction that supports HANDLE-based
/// file operations in addition to the standard file descriptor operations.
class WindowsFileAction : public FileAction {
public:
  WindowsFileAction() = default;

  /// Allow implicit conversion from a base FileAction. The Windows-specific
  /// handle fields default to INVALID_HANDLE_VALUE.
  WindowsFileAction(const FileAction &fa) : FileAction(fa) {}

  /// Reset this WindowsFileAction to its default state.
  void Clear() {
    FileAction::Clear();
    m_handle = LLDB_INVALID_PIPE;
    m_arg_handle = LLDB_INVALID_PIPE;
  }

  /// Configure this action to duplicate a Windows file handle.
  ///
  /// \param[in] fh
  ///     The source file handle to duplicate.
  /// \param[in] dup_fh
  ///     The target file handle.
  bool Duplicate(HANDLE fh, HANDLE dup_fh);

  /// Configure this action to associate a Windows file handle with a file.
  ///
  /// \param[in] fh
  ///     The file handle to use for the opened file.
  /// \param[in] file_spec
  ///     The file to open.
  /// \param[in] read
  ///     Open for reading.
  /// \param[in] write
  ///     Open for writing.
  bool Open(HANDLE fh, const FileSpec &file_spec, bool read, bool write);

  /// Get the Windows HANDLE for this action's file.
  ///
  /// If a HANDLE was stored directly, it is returned. Otherwise, the standard
  /// handles for STDIN/STDOUT/STDERR are returned based on the stored fd.
  HANDLE GetHandle() const;

  /// Get the Windows HANDLE argument for eFileActionDuplicate actions.
  HANDLE GetActionArgumentHandle() const;

private:
  HANDLE m_handle = LLDB_INVALID_PIPE;
  HANDLE m_arg_handle = LLDB_INVALID_PIPE;
};

} // namespace lldb_private

#endif // LLDB_HOST_WINDOWS_WINDOWSFILEACTION_H
