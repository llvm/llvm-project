//===-- PipePosix.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_POSIX_PIPEPOSIX_H
#define LLDB_HOST_POSIX_PIPEPOSIX_H

#include "lldb/Host/PipeBase.h"
#include <mutex>

namespace lldb_private {

/// \class PipePosix PipePosix.h "lldb/Host/posix/PipePosix.h"
/// A posix-based implementation of Pipe, a class that abtracts
///        unix style pipes.
///
/// A class that abstracts the LLDB core from host pipe functionality.
class PipePosix : public PipeBase {
public:
  static int kInvalidDescriptor;

  PipePosix();
  PipePosix(lldb::pipe_t read, lldb::pipe_t write);
  PipePosix(const PipePosix &) = delete;
  PipePosix(PipePosix &&pipe_posix);
  PipePosix &operator=(const PipePosix &) = delete;
  PipePosix &operator=(PipePosix &&pipe_posix);

  ~PipePosix() override;

  Status CreateNew(bool child_process_inherit) override;
  Status CreateNew(llvm::StringRef name, bool child_process_inherit) override;
  Status CreateWithUniqueName(llvm::StringRef prefix,
                              bool child_process_inherit,
                              llvm::SmallVectorImpl<char> &name) override;
  Status OpenAsReader(llvm::StringRef name,
                      bool child_process_inherit) override;
  llvm::Error OpenAsWriter(llvm::StringRef name, bool child_process_inherit,
                           const Timeout<std::micro> &timeout) override;

  bool CanRead() const override;
  bool CanWrite() const override;

  lldb::pipe_t GetReadPipe() const override {
    return lldb::pipe_t(GetReadFileDescriptor());
  }
  lldb::pipe_t GetWritePipe() const override {
    return lldb::pipe_t(GetWriteFileDescriptor());
  }

  int GetReadFileDescriptor() const override;
  int GetWriteFileDescriptor() const override;
  int ReleaseReadFileDescriptor() override;
  int ReleaseWriteFileDescriptor() override;
  void CloseReadFileDescriptor() override;
  void CloseWriteFileDescriptor() override;

  // Close both descriptors
  void Close() override;

  Status Delete(llvm::StringRef name) override;

  llvm::Expected<size_t>
  Write(const void *buf, size_t size,
        const Timeout<std::micro> &timeout = std::nullopt) override;

  llvm::Expected<size_t>
  Read(void *buf, size_t size,
       const Timeout<std::micro> &timeout = std::nullopt) override;

private:
  bool CanReadUnlocked() const;
  bool CanWriteUnlocked() const;

  int GetReadFileDescriptorUnlocked() const;
  int GetWriteFileDescriptorUnlocked() const;
  int ReleaseReadFileDescriptorUnlocked();
  int ReleaseWriteFileDescriptorUnlocked();
  void CloseReadFileDescriptorUnlocked();
  void CloseWriteFileDescriptorUnlocked();
  void CloseUnlocked();

  int m_fds[2];

  /// Mutexes for m_fds;
  mutable std::mutex m_read_mutex;
  mutable std::mutex m_write_mutex;
};

} // namespace lldb_private

#endif // LLDB_HOST_POSIX_PIPEPOSIX_H
