//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_POSIX_FILEPOSIX_H
#define LLDB_HOST_POSIX_FILEPOSIX_H

#include "lldb/Host/FileBase.h"

namespace lldb_private {

/// \class NativeFilePosix FilePosix.h "lldb/Host/posix/FilePosix.h"
/// POSIX implementation of NativeFile.
class NativeFilePosix : public NativeFileBase {
public:
  NativeFilePosix() = default;

  NativeFilePosix(FILE *fh, OpenOptions options, bool transfer_ownership);

  NativeFilePosix(int fd, OpenOptions options, bool transfer_ownership);

  // Bring the inherited single-argument Read/Write into scope so they aren't
  // hidden by the positional overloads declared below.
  using NativeFileBase::Read;
  using NativeFileBase::Write;

  Status GetFileSpec(FileSpec &file_spec) const override;
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

private:
  NativeFilePosix(const NativeFilePosix &) = delete;
  const NativeFilePosix &operator=(const NativeFilePosix &) = delete;
};

} // namespace lldb_private

#endif // LLDB_HOST_POSIX_FILEPOSIX_H
