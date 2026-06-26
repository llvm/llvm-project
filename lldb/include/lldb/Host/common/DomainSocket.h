//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_COMMON_DOMAINSOCKET_H
#define LLDB_HOST_COMMON_DOMAINSOCKET_H

#include "lldb/Host/Socket.h"
#include <string>
#include <vector>

namespace lldb_private {

/// Cross-platform AF_UNIX domain-socket logic.
///
/// Every operation on a domain socket (connect, listen, accept, name
/// lookup) is identical on POSIX and Windows, so it all lives here. The only
/// operation which differs between platforms is the CreatePair() factory: POSIX
/// has socketpair(2) while Windows must emulate it. That factory therefore
/// lives in the DomainSocketPosix / DomainSocketWindows implementation classes.
class DomainSocket : public Socket {
public:
  DomainSocket(NativeSocket socket, bool should_close);
  explicit DomainSocket(bool should_close);

  using Pair =
      std::pair<std::unique_ptr<DomainSocket>, std::unique_ptr<DomainSocket>>;

  Status Connect(llvm::StringRef name) override;
  Status Listen(llvm::StringRef name, int backlog) override;

  using Socket::Accept;
  llvm::Expected<std::vector<MainLoopBase::ReadHandleUP>>
  Accept(MainLoopBase &loop,
         std::function<void(std::unique_ptr<Socket> socket)> sock_cb) override;

  std::string GetRemoteConnectionURI() const override;

  std::vector<std::string> GetListeningConnectionURI() const override;

  static llvm::Expected<std::unique_ptr<DomainSocket>>
  FromBoundNativeSocket(NativeSocket sockfd, bool should_close);

protected:
  DomainSocket(SocketProtocol protocol);
  DomainSocket(SocketProtocol protocol, NativeSocket socket, bool should_close);

  virtual size_t GetNameOffset() const;
  virtual void DeleteSocketFile(llvm::StringRef name);
  std::string GetSocketName() const;

private:
  DomainSocket(NativeSocket socket, const DomainSocket &listen_socket);
};
} // namespace lldb_private

#endif // LLDB_HOST_COMMON_DOMAINSOCKET_H
