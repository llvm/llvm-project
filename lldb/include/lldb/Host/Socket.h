//===-- Socket.h ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Host_Socket_h_
#define liblldb_Host_Socket_h_

#include <memory>
#include <string>

#include "lldb/lldb-private.h"

#include "lldb/Host/IOObject.h"
#include "lldb/Host/Predicate.h"
#include "lldb/Host/SocketAddress.h"
#include "lldb/Utility/Error.h"

#ifdef _WIN32
#include "lldb/Host/windows/windows.h"
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

namespace llvm {
class StringRef;
}

namespace lldb_private {

#if defined(_MSC_VER)
typedef SOCKET NativeSocket;
#else
typedef int NativeSocket;
#endif

class Socket : public IOObject {
public:
  typedef enum {
    ProtocolTcp,
    ProtocolUdp,
    ProtocolUnixDomain,
    ProtocolUnixAbstract
  } SocketProtocol;

  static const NativeSocket kInvalidSocketValue;

  ~Socket() override;

  static std::unique_ptr<Socket> Create(const SocketProtocol protocol,
                                        bool child_processes_inherit,
                                        Error &error);

  virtual Error Connect(llvm::StringRef name) = 0;
  virtual Error Listen(llvm::StringRef name, int backlog) = 0;
  virtual Error Accept(llvm::StringRef name, bool child_processes_inherit,
                       Socket *&socket) = 0;

  // Initialize a Tcp Socket object in listening mode.  listen and accept are
  // implemented
  // separately because the caller may wish to manipulate or query the socket
  // after it is
  // initialized, but before entering a blocking accept.
  static Error TcpListen(llvm::StringRef host_and_port,
                         bool child_processes_inherit, Socket *&socket,
                         Predicate<uint16_t> *predicate, int backlog = 5);
  static Error TcpConnect(llvm::StringRef host_and_port,
                          bool child_processes_inherit, Socket *&socket);
  static Error UdpConnect(llvm::StringRef host_and_port,
                          bool child_processes_inherit, Socket *&send_socket,
                          Socket *&recv_socket);
  static Error UnixDomainConnect(llvm::StringRef host_and_port,
                                 bool child_processes_inherit, Socket *&socket);
  static Error UnixDomainAccept(llvm::StringRef host_and_port,
                                bool child_processes_inherit, Socket *&socket);
  static Error UnixAbstractConnect(llvm::StringRef host_and_port,
                                   bool child_processes_inherit,
                                   Socket *&socket);
  static Error UnixAbstractAccept(llvm::StringRef host_and_port,
                                  bool child_processes_inherit,
                                  Socket *&socket);

  int GetOption(int level, int option_name, int &option_value);
  int SetOption(int level, int option_name, int option_value);

  NativeSocket GetNativeSocket() const { return m_socket; }
  SocketProtocol GetSocketProtocol() const { return m_protocol; }

  Error Read(void *buf, size_t &num_bytes) override;
  Error Write(const void *buf, size_t &num_bytes) override;

  virtual Error PreDisconnect();
  Error Close() override;

  bool IsValid() const override { return m_socket != kInvalidSocketValue; }
  WaitableHandle GetWaitableHandle() override;

  static bool DecodeHostAndPort(llvm::StringRef host_and_port,
                                std::string &host_str, std::string &port_str,
                                int32_t &port, Error *error_ptr);

protected:
  Socket(NativeSocket socket, SocketProtocol protocol, bool should_close);

  virtual size_t Send(const void *buf, const size_t num_bytes);

  static void SetLastError(Error &error);
  static NativeSocket CreateSocket(const int domain, const int type,
                                   const int protocol,
                                   bool child_processes_inherit, Error &error);
  static NativeSocket AcceptSocket(NativeSocket sockfd, struct sockaddr *addr,
                                   socklen_t *addrlen,
                                   bool child_processes_inherit, Error &error);

  SocketProtocol m_protocol;
  NativeSocket m_socket;
};

} // namespace lldb_private

#endif // liblldb_Host_Socket_h_
