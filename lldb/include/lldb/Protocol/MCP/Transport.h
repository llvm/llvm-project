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
#include "llvm/ADT/StringRef.h"
#include <functional>
#include <string>

namespace lldb_protocol::mcp {

class Transport
    : public lldb_private::JSONRPCTransport<Request, Response, Notification> {
public:
  using LogCallback = std::function<void(llvm::StringRef message)>;

  Transport(lldb::IOObjectSP in, lldb::IOObjectSP out, std::string client_name,
            LogCallback log_callback = {});
  virtual ~Transport() = default;

  void Log(llvm::StringRef message) override;

private:
  std::string m_client_name;
  LogCallback m_log_callback;
};

} // namespace lldb_protocol::mcp

#endif
