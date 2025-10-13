//===-- UDPSocket.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_COMMON_UDPSOCKET_H
#define LLDB_HOST_COMMON_UDPSOCKET_H

#include "lldb/Host/Socket.h"

namespace lldb_private {
class UDPSocket : public Socket {
public:
  explicit UDPSocket(bool should_close);

  static llvm::Expected<std::unique_ptr<UDPSocket>>
  CreateConnected(llvm::StringRef name);

  std::string GetRemoteConnectionURI() const override;

private:
  UDPSocket(NativeSocket socket);

  size_t Send(const void *buf, const size_t num_bytes) override;
  Status Connect(llvm::StringRef name) override;
  Status Listen(llvm::StringRef name, int backlog) override;

  llvm::Expected<std::vector<MainLoopBase::ReadHandleUP>>
  Accept(MainLoopBase &loop,
         std::function<void(std::unique_ptr<Socket> socket)> sock_cb) override {
    return llvm::errorCodeToError(
        std::make_error_code(std::errc::operation_not_supported));
  }

  SocketAddress m_sockaddr;
};
}

#endif // LLDB_HOST_COMMON_UDPSOCKET_H
