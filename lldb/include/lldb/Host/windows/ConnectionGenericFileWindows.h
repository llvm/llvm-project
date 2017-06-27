//===-- ConnectionGenericFileWindows.h --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Host_windows_ConnectionGenericFileWindows_h_
#define liblldb_Host_windows_ConnectionGenericFileWindows_h_

#include "lldb/Host/windows/windows.h"
#include "lldb/Utility/Connection.h"
#include "lldb/lldb-types.h"

namespace lldb_private {

class Status;

class ConnectionGenericFile : public lldb_private::Connection {
public:
  ConnectionGenericFile();

  ConnectionGenericFile(lldb::file_t file, bool owns_file);

  ~ConnectionGenericFile() override;

  bool IsConnected() const override;

  lldb::ConnectionStatus Connect(llvm::StringRef s, Status *error_ptr) override;

  lldb::ConnectionStatus Disconnect(Status *error_ptr) override;

  size_t Read(void *dst, size_t dst_len, const Timeout<std::micro> &timeout,
              lldb::ConnectionStatus &status, Status *error_ptr) override;

  size_t Write(const void *src, size_t src_len, lldb::ConnectionStatus &status,
               Status *error_ptr) override;

  std::string GetURI() override;

  bool InterruptRead() override;

protected:
  OVERLAPPED m_overlapped;
  HANDLE m_file;
  HANDLE m_event_handles[2];
  bool m_owns_file;
  LARGE_INTEGER m_file_position;

  enum { kBytesAvailableEvent, kInterruptEvent };

private:
  void InitializeEventHandles();
  void IncrementFilePointer(DWORD amount);

  std::string m_uri;

  DISALLOW_COPY_AND_ASSIGN(ConnectionGenericFile);
};
}

#endif
