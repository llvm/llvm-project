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
#include <vector>

#include "lldb/Host/MainLoopBase.h"
#include "lldb/Utility/Timeout.h"
#include "lldb/lldb-private.h"

#include "lldb/Host/SocketAddress.h"
#include "lldb/Utility/IOObject.h"
#include "lldb/Utility/Status.h"

#ifdef _WIN32
#include "lldb/Host/Pipe.h"
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
typedef lldb::pipe_t shared_fd_t;
#else
typedef int NativeSocket;
typedef NativeSocket shared_fd_t;
#endif
class Socket;
class TCPSocket;
class UDPSocket;

class SharedSocket {
public:
  static const shared_fd_t kInvalidFD;

  SharedSocket(const Socket *socket, Status &error);

  shared_fd_t GetSendableFD() { return m_fd; }

  Status CompleteSending(lldb::pid_t child_pid);

  static Status GetNativeSocket(shared_fd_t fd, NativeSocket &socket);

private:
#ifdef _WIN32
  Pipe m_socket_pipe;
  NativeSocket m_socket;
#endif
  shared_fd_t m_fd;
};

class Socket : public IOObject {
public:
  enum SocketProtocol {
    ProtocolTcp,
    ProtocolUdp,
    ProtocolUnixDomain,
    ProtocolUnixAbstract
  };

  struct HostAndPort {
    std::string hostname;
    uint16_t port;

    bool operator==(const HostAndPort &R) const {
      return port == R.port && hostname == R.hostname;
    }
  };

  static const NativeSocket kInvalidSocketValue;

  ~Socket() override;

  static const char *FindSchemeByProtocol(const SocketProtocol protocol);
  static bool FindProtocolByScheme(const char *scheme,
                                   SocketProtocol &protocol);

  static llvm::Error Initialize();
  static void Terminate();

  static std::unique_ptr<Socket> Create(const SocketProtocol protocol,
                                        Status &error);

  virtual Status Connect(llvm::StringRef name) = 0;
  virtual Status Listen(llvm::StringRef name, int backlog) = 0;

  // Use the provided main loop instance to accept new connections. The callback
  // will be called (from MainLoop::Run) for each new connection. This function
  // does not block.
  virtual llvm::Expected<std::vector<MainLoopBase::ReadHandleUP>>
  Accept(MainLoopBase &loop,
         std::function<void(std::unique_ptr<Socket> socket)> sock_cb) = 0;

  // Accept a single connection and "return" it in the pointer argument. This
  // function blocks until the connection arrives.
  virtual Status Accept(const Timeout<std::micro> &timeout, Socket *&socket);

  // Initialize a Tcp Socket object in listening mode.  listen and accept are
  // implemented separately because the caller may wish to manipulate or query
  // the socket after it is initialized, but before entering a blocking accept.
  static llvm::Expected<std::unique_ptr<TCPSocket>>
  TcpListen(llvm::StringRef host_and_port, int backlog = 5);

  static llvm::Expected<std::unique_ptr<Socket>>
  TcpConnect(llvm::StringRef host_and_port);

  static llvm::Expected<std::unique_ptr<UDPSocket>>
  UdpConnect(llvm::StringRef host_and_port);

  static int GetOption(NativeSocket sockfd, int level, int option_name,
                       int &option_value);
  int GetOption(int level, int option_name, int &option_value) {
    return GetOption(m_socket, level, option_name, option_value);
  };

  static int SetOption(NativeSocket sockfd, int level, int option_name,
                       int option_value);
  int SetOption(int level, int option_name, int option_value) {
    return SetOption(m_socket, level, option_name, option_value);
  };

  NativeSocket GetNativeSocket() const { return m_socket; }
  SocketProtocol GetSocketProtocol() const { return m_protocol; }

  Status Read(void *buf, size_t &num_bytes) override;
  Status Write(const void *buf, size_t &num_bytes) override;

  Status Close() override;

  bool IsValid() const override { return m_socket != kInvalidSocketValue; }
  WaitableHandle GetWaitableHandle() override;

  static llvm::Expected<HostAndPort>
  DecodeHostAndPort(llvm::StringRef host_and_port);

  // If this Socket is connected then return the URI used to connect.
  virtual std::string GetRemoteConnectionURI() const { return ""; };

  // If the Socket is listening then return the URI for clients to connect.
  virtual std::vector<std::string> GetListeningConnectionURI() const {
    return {};
  }

protected:
  Socket(SocketProtocol protocol, bool should_close);

  virtual size_t Send(const void *buf, const size_t num_bytes);

  static int CloseSocket(NativeSocket sockfd);
  static Status GetLastError();
  static void SetLastError(Status &error);
  static NativeSocket CreateSocket(const int domain, const int type,
                                   const int protocol, Status &error);
  static NativeSocket AcceptSocket(NativeSocket sockfd, struct sockaddr *addr,
                                   socklen_t *addrlen, Status &error);

  SocketProtocol m_protocol;
  NativeSocket m_socket;
  bool m_should_close_fd;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const Socket::HostAndPort &HP);

} // namespace lldb_private

#endif // LLDB_HOST_SOCKET_H
