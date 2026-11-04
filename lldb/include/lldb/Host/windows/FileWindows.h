//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_WINDOWS_FILEWINDOWS_H
#define LLDB_HOST_WINDOWS_FILEWINDOWS_H

#include "lldb/Host/FileBase.h"

namespace lldb_private {

/// \class NativeFileWindows FileWindows.h "lldb/Host/windows/FileWindows.h"
/// Windows implementation of NativeFile.
class NativeFileWindows : public NativeFileBase {
public:
  NativeFileWindows() = default;

  NativeFileWindows(FILE *fh, OpenOptions options, bool transfer_ownership);

  NativeFileWindows(int fd, OpenOptions options, bool transfer_ownership);

  // Bring the inherited single-argument Read/Write into scope so they aren't
  // hidden by the positional overloads declared below.
  using NativeFileBase::Read;
  using NativeFileBase::Write;

  WaitableHandle GetWaitableHandle() override;
  Status Sync() override;
  Status Read(void *dst, size_t &num_bytes, off_t &offset) override;
  Status Write(const void *src, size_t &num_bytes, off_t &offset) override;

  static char ID;
  bool isA(const void *classID) const override {
    return classID == &ID || NativeFileBase::isA(classID);
  }
  static bool classof(const File *file) { return file->isA(&ID); }

protected:
  void CalculateInteractiveAndTerminal() override;

  int Fileno(FILE *fh) const override;
  int Dup(int fd) const override;

  bool TryWriteStreamUnlocked(const void *buf, size_t &num_bytes,
                              Status &error) override;

  void OnStreamOpened() override;

private:
  /// Set when this file wraps stdin/stdout/stderr connected to a console;
  /// triggers the raw_fd_ostream path for correct non-ASCII output.
  bool m_is_windows_console = false;

  NativeFileWindows(const NativeFileWindows &) = delete;
  const NativeFileWindows &operator=(const NativeFileWindows &) = delete;
};

} // namespace lldb_private

#endif // LLDB_HOST_WINDOWS_FILEWINDOWS_H
