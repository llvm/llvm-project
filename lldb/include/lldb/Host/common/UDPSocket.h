//===-- UDPSocket.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_UDPSocket_h_
#define liblldb_UDPSocket_h_

#include "lldb/Host/Socket.h"

namespace lldb_private {
class UDPSocket : public Socket {
public:
  UDPSocket(bool should_close, bool child_processes_inherit);

  static Status Connect(llvm::StringRef name, bool child_processes_inherit,
                        Socket *&socket);

private:
  UDPSocket(NativeSocket socket);

  size_t Send(const void *buf, const size_t num_bytes) override;
  Status Connect(llvm::StringRef name) override;
  Status Listen(llvm::StringRef name, int backlog) override;
  Status Accept(Socket *&socket) override;

  SocketAddress m_sockaddr;
};
}

#endif // ifndef liblldb_UDPSocket_h_
