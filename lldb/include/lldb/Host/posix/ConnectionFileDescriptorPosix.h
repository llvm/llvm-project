//===-- ConnectionFileDescriptorPosix.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_POSIX_CONNECTIONFILEDESCRIPTORPOSIX_H
#define LLDB_HOST_POSIX_CONNECTIONFILEDESCRIPTORPOSIX_H

#include <atomic>
#include <memory>
#include <mutex>

#include "lldb/lldb-forward.h"

#include "lldb/Host/Pipe.h"
#include "lldb/Utility/Connection.h"
#include "lldb/Utility/IOObject.h"
#include "lldb/Utility/Predicate.h"

namespace lldb_private {

class Status;
class Socket;
class SocketAddress;

class ConnectionFileDescriptor : public Connection {
public:
  ConnectionFileDescriptor(bool child_processes_inherit = false);

  ConnectionFileDescriptor(int fd, bool owns_fd);

  ConnectionFileDescriptor(Socket *socket);

  ~ConnectionFileDescriptor() override;

  bool IsConnected() const override;

  lldb::ConnectionStatus Connect(llvm::StringRef s, Status *error_ptr) override;

  lldb::ConnectionStatus Disconnect(Status *error_ptr) override;

  size_t Read(void *dst, size_t dst_len, const Timeout<std::micro> &timeout,
              lldb::ConnectionStatus &status, Status *error_ptr) override;

  size_t Write(const void *src, size_t src_len, lldb::ConnectionStatus &status,
               Status *error_ptr) override;

  std::string GetURI() override;

  lldb::ConnectionStatus BytesAvailable(const Timeout<std::micro> &timeout,
                                        Status *error_ptr);

  bool InterruptRead() override;

  lldb::IOObjectSP GetReadObject() override { return m_io_sp; }

  uint16_t GetListeningPort(const Timeout<std::micro> &timeout);

  bool GetChildProcessesInherit() const;
  void SetChildProcessesInherit(bool child_processes_inherit);

protected:
  void OpenCommandPipe();

  void CloseCommandPipe();

  lldb::ConnectionStatus SocketListenAndAccept(llvm::StringRef host_and_port,
                                               Status *error_ptr);

  lldb::ConnectionStatus ConnectTCP(llvm::StringRef host_and_port,
                                    Status *error_ptr);

  lldb::ConnectionStatus ConnectUDP(llvm::StringRef args, Status *error_ptr);

  lldb::ConnectionStatus NamedSocketConnect(llvm::StringRef socket_name,
                                            Status *error_ptr);

  lldb::ConnectionStatus NamedSocketAccept(llvm::StringRef socket_name,
                                           Status *error_ptr);

  lldb::ConnectionStatus UnixAbstractSocketConnect(llvm::StringRef socket_name,
                                                   Status *error_ptr);

  lldb::ConnectionStatus ConnectFD(llvm::StringRef args, Status *error_ptr);

  lldb::ConnectionStatus ConnectFile(llvm::StringRef args, Status *error_ptr);

  lldb::ConnectionStatus ConnectSerialPort(llvm::StringRef args,
                                           Status *error_ptr);

  lldb::IOObjectSP m_io_sp;

  Predicate<uint16_t>
      m_port_predicate; // Used when binding to port zero to wait for the thread
                        // that creates the socket, binds and listens to
                        // resolve the port number.

  Pipe m_pipe;
  std::recursive_mutex m_mutex;
  std::atomic<bool> m_shutting_down; // This marks that we are shutting down so
                                     // if we get woken up from
  // BytesAvailable to disconnect, we won't try to read again.
  bool m_waiting_for_accept = false;
  bool m_child_processes_inherit;

  std::string m_uri;

private:
  void InitializeSocket(Socket *socket);

  ConnectionFileDescriptor(const ConnectionFileDescriptor &) = delete;
  const ConnectionFileDescriptor &
  operator=(const ConnectionFileDescriptor &) = delete;
};

} // namespace lldb_private

#endif // LLDB_HOST_POSIX_CONNECTIONFILEDESCRIPTORPOSIX_H
