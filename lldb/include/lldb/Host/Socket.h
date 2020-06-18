//===-- Socket.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_SOCKET_H
#define LLDB_HOST_SOCKET_H

#include <memory>
#include <string>

#include "lldb/lldb-private.h"

#include "lldb/Host/SocketAddress.h"
#include "lldb/Utility/IOObject.h"
#include "lldb/Utility/Predicate.h"
#include "lldb/Utility/Status.h"

#ifdef _WIN32
#include "lldb/Host/windows/windows.h"
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

namespace llvm {
class StringRef;
}

namespace lldb_private {

#if defined(_WIN32)
typedef SOCKET NativeSocket;
#else
typedef int NativeSocket;
#endif
class TCPSocket;
class UDPSocket;

class Socket : public IOObject {
public:
  enum SocketProtocol {
    ProtocolTcp,
    ProtocolUdp,
    ProtocolUnixDomain,
    ProtocolUnixAbstract
  };

  static const NativeSocket kInvalidSocketValue;

  ~Socket() override;

  static llvm::Error Initialize();
  static void Terminate();

  static std::unique_ptr<Socket> Create(const SocketProtocol protocol,
                                        bool child_processes_inherit,
                                        Status &error);

  virtual Status Connect(llvm::StringRef name) = 0;
  virtual Status Listen(llvm::StringRef name, int backlog) = 0;
  virtual Status Accept(Socket *&socket) = 0;

  // Initialize a Tcp Socket object in listening mode.  listen and accept are
  // implemented separately because the caller may wish to manipulate or query
  // the socket after it is initialized, but before entering a blocking accept.
  static llvm::Expected<std::unique_ptr<TCPSocket>>
  TcpListen(llvm::StringRef host_and_port, bool child_processes_inherit,
            Predicate<uint16_t> *predicate, int backlog = 5);

  static llvm::Expected<std::unique_ptr<Socket>>
  TcpConnect(llvm::StringRef host_and_port, bool child_processes_inherit);

  static llvm::Expected<std::unique_ptr<UDPSocket>>
  UdpConnect(llvm::StringRef host_and_port, bool child_processes_inherit);

  static Status UnixDomainConnect(llvm::StringRef host_and_port,
                                  bool child_processes_inherit,
                                  Socket *&socket);
  static Status UnixDomainAccept(llvm::StringRef host_and_port,
                                 bool child_processes_inherit, Socket *&socket);
  static Status UnixAbstractConnect(llvm::StringRef host_and_port,
                                    bool child_processes_inherit,
                                    Socket *&socket);
  static Status UnixAbstractAccept(llvm::StringRef host_and_port,
                                   bool child_processes_inherit,
                                   Socket *&socket);

  int GetOption(int level, int option_name, int &option_value);
  int SetOption(int level, int option_name, int option_value);

  NativeSocket GetNativeSocket() const { return m_socket; }
  SocketProtocol GetSocketProtocol() const { return m_protocol; }

  Status Read(void *buf, size_t &num_bytes) override;
  Status Write(const void *buf, size_t &num_bytes) override;

  virtual Status PreDisconnect();
  Status Close() override;

  bool IsValid() const override { return m_socket != kInvalidSocketValue; }
  WaitableHandle GetWaitableHandle() override;

  static bool DecodeHostAndPort(llvm::StringRef host_and_port,
                                std::string &host_str, std::string &port_str,
                                int32_t &port, Status *error_ptr);

  // If this Socket is connected then return the URI used to connect.
  virtual std::string GetRemoteConnectionURI() const { return ""; };

protected:
  Socket(SocketProtocol protocol, bool should_close,
         bool m_child_process_inherit);

  virtual size_t Send(const void *buf, const size_t num_bytes);

  static void SetLastError(Status &error);
  static NativeSocket CreateSocket(const int domain, const int type,
                                   const int protocol,
                                   bool child_processes_inherit, Status &error);
  static NativeSocket AcceptSocket(NativeSocket sockfd, struct sockaddr *addr,
                                   socklen_t *addrlen,
                                   bool child_processes_inherit, Status &error);

  SocketProtocol m_protocol;
  NativeSocket m_socket;
  bool m_child_processes_inherit;
  bool m_should_close_fd;
};

} // namespace lldb_private

#endif // LLDB_HOST_SOCKET_H
