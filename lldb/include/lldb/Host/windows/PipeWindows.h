//===-- PipePosix.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Host_windows_PipeWindows_h_
#define liblldb_Host_windows_PipeWindows_h_

#include "lldb/Host/PipeBase.h"
#include "lldb/Host/windows/windows.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class Pipe PipeWindows.h "lldb/Host/windows/PipeWindows.h"
/// @brief A windows-based implementation of Pipe, a class that abtracts
///        unix style pipes.
///
/// A class that abstracts the LLDB core from host pipe functionality.
//----------------------------------------------------------------------
class PipeWindows : public PipeBase {
public:
  PipeWindows();
  ~PipeWindows() override;

  Error CreateNew(bool child_process_inherit) override;
  Error CreateNew(llvm::StringRef name, bool child_process_inherit) override;
  Error CreateWithUniqueName(llvm::StringRef prefix, bool child_process_inherit,
                             llvm::SmallVectorImpl<char> &name) override;
  Error OpenAsReader(llvm::StringRef name, bool child_process_inherit) override;
  Error
  OpenAsWriterWithTimeout(llvm::StringRef name, bool child_process_inherit,
                          const std::chrono::microseconds &timeout) override;

  bool CanRead() const override;
  bool CanWrite() const override;

  int GetReadFileDescriptor() const override;
  int GetWriteFileDescriptor() const override;
  int ReleaseReadFileDescriptor() override;
  int ReleaseWriteFileDescriptor() override;
  void CloseReadFileDescriptor() override;
  void CloseWriteFileDescriptor() override;

  void Close() override;

  Error Delete(llvm::StringRef name) override;

  Error Write(const void *buf, size_t size, size_t &bytes_written) override;
  Error ReadWithTimeout(void *buf, size_t size,
                        const std::chrono::microseconds &timeout,
                        size_t &bytes_read) override;

  // PipeWindows specific methods.  These allow access to the underlying OS
  // handle.
  HANDLE GetReadNativeHandle();
  HANDLE GetWriteNativeHandle();

private:
  Error OpenNamedPipe(llvm::StringRef name, bool child_process_inherit,
                      bool is_read);

  HANDLE m_read;
  HANDLE m_write;

  int m_read_fd;
  int m_write_fd;

  OVERLAPPED m_read_overlapped;
  OVERLAPPED m_write_overlapped;
};

} // namespace lldb_private

#endif // liblldb_Host_posix_PipePosix_h_
