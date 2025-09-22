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
#include "lldb/Protocol/MCP/Protocol.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringRef.h"

namespace lldb_protocol::mcp {

/// Generic transport that uses the MCP protocol.
using MCPTransport = lldb_private::Transport<Request, Response, Notification>;

/// Generic logging callback, to allow the MCP server / client / transport layer
/// to be independent of the lldb log implementation.
using LogCallback = llvm::unique_function<void(llvm::StringRef message)>;

class Transport final
    : public lldb_private::JSONRPCTransport<Request, Response, Notification> {
public:
  Transport(lldb::IOObjectSP in, lldb::IOObjectSP out,
            LogCallback log_callback = {});
  virtual ~Transport() = default;

  /// Transport is not copyable.
  /// @{
  Transport(const Transport &) = delete;
  void operator=(const Transport &) = delete;
  /// @}

  void Log(llvm::StringRef message) override;

private:
  LogCallback m_log_callback;
};

} // namespace lldb_protocol::mcp

#endif
