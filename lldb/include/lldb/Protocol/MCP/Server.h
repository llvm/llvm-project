//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PROTOCOL_MCP_SERVER_H
#define LLDB_PROTOCOL_MCP_SERVER_H

#include "lldb/Host/JSONTransport.h"
#include "lldb/Host/MainLoop.h"
#include "lldb/Protocol/MCP/Protocol.h"
#include "lldb/Protocol/MCP/Resource.h"
#include "lldb/Protocol/MCP/Tool.h"
#include "lldb/Protocol/MCP/Transport.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace lldb_protocol::mcp {

/// Information about this instance of lldb's MCP server for lldb-mcp to use to
/// coordinate connecting an lldb-mcp client.
struct ServerInfo {
  std::string connection_uri;
  lldb::pid_t pid;
};
llvm::json::Value toJSON(const ServerInfo &);
bool fromJSON(const llvm::json::Value &, ServerInfo &, llvm::json::Path);

class Server : public MCPTransport::MessageHandler {
public:
  Server(std::string name, std::string version,
         std::unique_ptr<MCPTransport> transport_up,
         lldb_private::MainLoop &loop);
  ~Server() = default;

  using NotificationHandler = std::function<void(const Notification &)>;

  void AddTool(std::unique_ptr<Tool> tool);
  void AddResourceProvider(std::unique_ptr<ResourceProvider> resource_provider);
  void AddNotificationHandler(llvm::StringRef method,
                              NotificationHandler handler);

  llvm::Error Run();

protected:
  ServerCapabilities GetCapabilities();

  using RequestHandler =
      std::function<llvm::Expected<Response>(const Request &)>;

  void AddRequestHandlers();

  void AddRequestHandler(llvm::StringRef method, RequestHandler handler);

  llvm::Expected<std::optional<Message>> HandleData(llvm::StringRef data);

  llvm::Expected<Response> Handle(const Request &request);
  void Handle(const Notification &notification);

  llvm::Expected<Response> InitializeHandler(const Request &);

  llvm::Expected<Response> ToolsListHandler(const Request &);
  llvm::Expected<Response> ToolsCallHandler(const Request &);

  llvm::Expected<Response> ResourcesListHandler(const Request &);
  llvm::Expected<Response> ResourcesReadHandler(const Request &);

  void Received(const Request &) override;
  void Received(const Response &) override;
  void Received(const Notification &) override;
  void OnError(llvm::Error) override;
  void OnClosed() override;

  void TerminateLoop();

private:
  const std::string m_name;
  const std::string m_version;

  std::unique_ptr<MCPTransport> m_transport_up;
  lldb_private::MainLoop &m_loop;

  llvm::StringMap<std::unique_ptr<Tool>> m_tools;
  std::vector<std::unique_ptr<ResourceProvider>> m_resource_providers;

  llvm::StringMap<RequestHandler> m_request_handlers;
  llvm::StringMap<NotificationHandler> m_notification_handlers;
};

} // namespace lldb_protocol::mcp

#endif
