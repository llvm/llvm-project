//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PROTOCOL_MCP_SERVER_H
#define LLDB_PROTOCOL_MCP_SERVER_H

#include "lldb/Protocol/MCP/Protocol.h"
#include "lldb/Protocol/MCP/Resource.h"
#include "lldb/Protocol/MCP/Tool.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include <mutex>

namespace lldb_protocol::mcp {

class Server {
public:
  Server(std::string name, std::string version);
  virtual ~Server() = default;

  void AddTool(std::unique_ptr<Tool> tool);
  void AddResourceProvider(std::unique_ptr<ResourceProvider> resource_provider);

protected:
  virtual Capabilities GetCapabilities() = 0;

  using RequestHandler =
      std::function<llvm::Expected<Response>(const Request &)>;
  using NotificationHandler = std::function<void(const Notification &)>;

  void AddRequestHandlers();

  void AddRequestHandler(llvm::StringRef method, RequestHandler handler);
  void AddNotificationHandler(llvm::StringRef method,
                              NotificationHandler handler);

  llvm::Expected<std::optional<Message>> HandleData(llvm::StringRef data);

  llvm::Expected<Response> Handle(Request request);
  void Handle(Notification notification);

  llvm::Expected<Response> InitializeHandler(const Request &);

  llvm::Expected<Response> ToolsListHandler(const Request &);
  llvm::Expected<Response> ToolsCallHandler(const Request &);

  llvm::Expected<Response> ResourcesListHandler(const Request &);
  llvm::Expected<Response> ResourcesReadHandler(const Request &);

  std::mutex m_mutex;

private:
  const std::string m_name;
  const std::string m_version;

  llvm::StringMap<std::unique_ptr<Tool>> m_tools;
  std::vector<std::unique_ptr<ResourceProvider>> m_resource_providers;

  llvm::StringMap<RequestHandler> m_request_handlers;
  llvm::StringMap<NotificationHandler> m_notification_handlers;
};

} // namespace lldb_protocol::mcp

#endif
