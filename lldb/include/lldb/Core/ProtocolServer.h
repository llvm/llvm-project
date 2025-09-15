//===-- ProtocolServer.h --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_CORE_PROTOCOLSERVER_H
#define LLDB_CORE_PROTOCOLSERVER_H

#include "lldb/Core/PluginInterface.h"
#include "lldb/Host/Socket.h"
#include "lldb/lldb-private-interfaces.h"

namespace lldb_private {

class ProtocolServer : public PluginInterface {
public:
  ProtocolServer() = default;
  virtual ~ProtocolServer() = default;

  static ProtocolServer *GetOrCreate(llvm::StringRef name);

  static llvm::Error Terminate();

  static std::vector<llvm::StringRef> GetSupportedProtocols();

  struct Connection {
    Socket::SocketProtocol protocol;
    std::string name;
  };

  virtual llvm::Error Start(Connection connection) = 0;
  virtual llvm::Error Stop() = 0;

  virtual Socket *GetSocket() const = 0;
};

} // namespace lldb_private

#endif
