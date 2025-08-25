//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PROTOCOL_MCP_SERVER_H
#define LLDB_PROTOCOL_MCP_SERVER_H

#include "lldb/Host/MainLoop.h"
#include "lldb/Protocol/MCP/Binder.h"
#include "lldb/Protocol/MCP/Protocol.h"
#include "lldb/Protocol/MCP/Resource.h"
#include "lldb/Protocol/MCP/Tool.h"
#include "lldb/Protocol/MCP/Transport.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include <memory>
#include <mutex>

namespace lldb_private::mcp {
class ProtocolServerMCP;
} // namespace lldb_private::mcp

namespace lldb_protocol::mcp {

class Server {
  friend class lldb_private::mcp::ProtocolServerMCP;
  friend class lldb_private::mcp::ProtocolServerMCP;

public:
  Server(std::string name, std::string version,
         std::unique_ptr<MCPTransport> transport_up,
         lldb_private::MainLoop &loop);
  ~Server() = default;

  void AddTool(std::unique_ptr<Tool> tool);
  void AddResourceProvider(std::unique_ptr<ResourceProvider> resource_provider);

  llvm::Error Run();

  Binder &GetBinder() { return m_binder; };

protected:
  ServerCapabilities GetCapabilities();

  llvm::Expected<InitializeResult>
  InitializeHandler(const InitializeParams &request);

  llvm::Expected<ToolsListResult> ToolsListHandler(const Void &);
  llvm::Expected<ToolsCallResult> ToolsCallHandler(const ToolsCallParams &);

  llvm::Expected<ResourcesListResult> ResourcesListHandler(const Void &);
  llvm::Expected<ResourcesReadResult>
  ResourcesReadHandler(const ResourcesReadParams &);

  void TerminateLoop();

  std::mutex m_mutex;

private:
  const std::string m_name;
  const std::string m_version;

  std::unique_ptr<MCPTransport> m_transport_up;
  lldb_private::MainLoop &m_loop;

  llvm::StringMap<std::unique_ptr<Tool>> m_tools;
  std::vector<std::unique_ptr<ResourceProvider>> m_resource_providers;
  Binder m_binder;
};

} // namespace lldb_protocol::mcp

#endif
