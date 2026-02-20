//===-- FileAction.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_FILEACTION_H
#define LLDB_HOST_FILEACTION_H

#include "lldb/Utility/FileSpec.h"
#include "lldb/lldb-types.h"
#include <string>
#include <variant>

namespace lldb_private {

/// Represents a file descriptor action to be performed during process launch.
///
/// FileAction encapsulates operations like opening, closing, or duplicating
/// file descriptors that should be applied when spawning a new process.
class FileAction {
public:
  enum Action {
    eFileActionNone,
    eFileActionClose,
    eFileActionDuplicate,
    eFileActionOpen
  };

  FileAction();

  /// Reset this FileAction to its default state.
  void Clear();

  /// Configure this action to close a file descriptor.
  bool Close(int fd);

  /// Configure this action to duplicate a file descriptor.
  ///
  /// \param[in] fd
  ///     The file descriptor to duplicate.
  /// \param[in] dup_file
  ///     The target file descriptor number.
  bool Duplicate(int fd, int dup_file);

#ifdef _WIN32
  /// Configure this action to duplicate a file handle.
  ///
  /// \param[in] fd
  ///     The file descriptor to duplicate.
  /// \param[in] dup_file
  ///     The target file descriptor number.
  bool Duplicate(void *fh, void *dup_fh);
#endif

  /// Configure this action to open a file.
  ///
  /// \param[in] fd
  ///     The file descriptor to use for the opened file.
  /// \param[in] file_spec
  ///     The file to open.
  /// \param[in] read
  ///     Open for reading.
  /// \param[in] write
  ///     Open for writing.
  bool Open(int fd, const FileSpec &file_spec, bool read, bool write);

#ifdef _WIN32
  /// Configure this action to open a file (Windows handle version).
  ///
  /// This method will open a CRT file descriptor to the handle and
  /// store that descriptor internally.
  ///
  /// \param[in] fh
  ///     The file handle to use for the opened file.
  /// \param[in] file_spec
  ///     The file to open.
  /// \param[in] read
  ///     Open for reading.
  /// \param[in] write
  ///     Open for writing.
  bool Open(void *fh, const FileSpec &file_spec, bool read, bool write);
#endif

  /// Get the file descriptor this action applies to.
  int GetFD() const;

#ifdef _WIN32
  /// Get the Windows handle for this file descriptor.
  ///
  /// The handle is converted from the file descriptor which is stored
  /// internally. The initial file descriptor must have been registered in the
  /// CRT before.
  void *GetHandle() const;
#endif

  /// Get the type of action.
  Action GetAction() const { return m_action; }

#ifdef _WIN32
  /// Get the file handle argument for eFileActionDuplicate actions.
  void *GetActionArgumentHandle() const;
#endif

  /// Get the action-specific argument.
  ///
  /// For eFileActionOpen, returns the open flags (O_RDONLY, etc.).
  /// For eFileActionDuplicate, returns the target fd to duplicate to.
  int GetActionArgument() const;

  /// Get the file specification for open actions.
  const FileSpec &GetFileSpec() const;

  void Dump(Stream &stream) const;

protected:
  /// The action for this file.
  Action m_action = eFileActionNone;
  /// An existing file descriptor.
  std::variant<int, void *> m_file = -1;
  /// oflag for eFileActionOpen, dup_fd for eFileActionDuplicate.
  std::variant<int, void *> m_arg = -1;
  /// File spec to use for opening after fork or posix_spawn.
  FileSpec m_file_spec;
};

} // namespace lldb_private

#endif
