//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PROTOCOL_MCP_TRANSPORT_H
#define LLDB_PROTOCOL_MCP_TRANSPORT_H

#include "lldb/Host/JSONTransport.h"
#include "lldb/Host/Socket.h"
#include "lldb/Protocol/MCP/Protocol.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

namespace lldb_protocol::mcp {

using MCPTransport = lldb_private::Transport<Request, Response, Notification>;
using MCPTransportUP = std::unique_ptr<MCPTransport>;

llvm::StringRef CommunicationSocketPath();
llvm::Expected<lldb::IOObjectSP> Connect();

class Transport final
    : public lldb_private::JSONRPCTransport<Request, Response, Notification> {
public:
  using LogCallback = std::function<void(llvm::StringRef message)>;

  Transport(lldb::IOObjectSP input, lldb::IOObjectSP output,
            std::string client_name = "", LogCallback log_callback = {});

  void Log(llvm::StringRef message) override;

  static llvm::Expected<MCPTransportUP>
  Connect(llvm::raw_ostream *logger = nullptr);

private:
  std::string m_client_name;
  LogCallback m_log_callback;
};
using TransportUP = std::unique_ptr<Transport>;

} // namespace lldb_protocol::mcp

#endif
