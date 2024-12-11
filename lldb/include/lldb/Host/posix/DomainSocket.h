//===-- DomainSocket.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_POSIX_DOMAINSOCKET_H
#define LLDB_HOST_POSIX_DOMAINSOCKET_H

#include "lldb/Host/Socket.h"

namespace lldb_private {
class DomainSocket : public Socket {
public:
  DomainSocket(NativeSocket socket, bool should_close,
               bool child_processes_inherit);
  DomainSocket(bool should_close, bool child_processes_inherit);

  Status Connect(llvm::StringRef name) override;
  Status Listen(llvm::StringRef name, int backlog) override;

  using Socket::Accept;
  llvm::Expected<std::vector<MainLoopBase::ReadHandleUP>>
  Accept(MainLoopBase &loop,
         std::function<void(std::unique_ptr<Socket> socket)> sock_cb) override;

  std::string GetRemoteConnectionURI() const override;

protected:
  DomainSocket(SocketProtocol protocol, bool child_processes_inherit);

  virtual size_t GetNameOffset() const;
  virtual void DeleteSocketFile(llvm::StringRef name);
  std::string GetSocketName() const;

private:
  DomainSocket(NativeSocket socket, const DomainSocket &listen_socket);
};
}

#endif // LLDB_HOST_POSIX_DOMAINSOCKET_H
