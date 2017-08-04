//===-- TCPSocket.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_TCPSocket_h_
#define liblldb_TCPSocket_h_

#include "lldb/Host/Socket.h"

namespace lldb_private {
class TCPSocket : public Socket {
public:
  TCPSocket(NativeSocket socket, bool should_close);
  TCPSocket(bool child_processes_inherit, Error &error);

  // returns port number or 0 if error
  uint16_t GetLocalPortNumber() const;

  // returns ip address string or empty string if error
  std::string GetLocalIPAddress() const;

  // must be connected
  // returns port number or 0 if error
  uint16_t GetRemotePortNumber() const;

  // must be connected
  // returns ip address string or empty string if error
  std::string GetRemoteIPAddress() const;

  int SetOptionNoDelay();
  int SetOptionReuseAddress();

  Error Connect(llvm::StringRef name) override;
  Error Listen(llvm::StringRef name, int backlog) override;
  Error Accept(llvm::StringRef name, bool child_processes_inherit,
               Socket *&conn_socket) override;
};
}

#endif // ifndef liblldb_TCPSocket_h_
