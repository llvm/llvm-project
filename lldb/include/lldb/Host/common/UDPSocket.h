//===-- UDPSocket.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_UDPSocket_h_
#define liblldb_UDPSocket_h_

#include "lldb/Host/Socket.h"

namespace lldb_private {
class UDPSocket : public Socket {
public:
  UDPSocket(bool should_close, bool child_processes_inherit);

  static Error Connect(llvm::StringRef name, bool child_processes_inherit,
                       Socket *&socket);

private:
  UDPSocket(NativeSocket socket, const UDPSocket &listen_socket);

  size_t Send(const void *buf, const size_t num_bytes) override;
  Error Connect(llvm::StringRef name) override;
  Error Listen(llvm::StringRef name, int backlog) override;
  Error Accept(Socket *&socket) override;

  Error CreateSocket();

  SocketAddress m_sockaddr;
};
}

#endif // ifndef liblldb_UDPSocket_h_
