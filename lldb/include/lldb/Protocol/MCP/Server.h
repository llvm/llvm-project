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
#include "lldb/Protocol/MCP/Protocol.h"
#include "lldb/Protocol/MCP/Resource.h"
#include "lldb/Protocol/MCP/Tool.h"
#include "lldb/Protocol/MCP/Transport.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Signals.h"
#include <memory>
#include <string>
#include <vector>

namespace lldb_protocol::mcp {

class Server {

  using MCPTransportUP = std::unique_ptr<lldb_protocol::mcp::MCPTransport>;

  using ReadHandleUP = lldb_private::MainLoop::ReadHandleUP;

public:
  Server(std::string name, std::string version, LogCallback log_callback = {});
  ~Server() = default;

  void AddTool(std::unique_ptr<Tool> tool);
  void AddResourceProvider(std::unique_ptr<ResourceProvider> resource_provider);

  llvm::Error Accept(lldb_private::MainLoop &, MCPTransportUP);

protected:
  MCPBinderUP Bind(MCPTransport &);

  ServerCapabilities GetCapabilities();

  llvm::Expected<InitializeResult> InitializeHandler(const InitializeParams &);

  llvm::Expected<ListToolsResult> ToolsListHandler();
  llvm::Expected<CallToolResult> ToolsCallHandler(const CallToolParams &);

  llvm::Expected<ListResourcesResult> ResourcesListHandler();
  llvm::Expected<ReadResourceResult>
  ResourcesReadHandler(const ReadResourceParams &);

  template <typename... Ts> inline auto Logv(const char *Fmt, Ts &&...Vals) {
    Log(llvm::formatv(Fmt, std::forward<Ts>(Vals)...).str());
  }
  void Log(llvm::StringRef message) {
    if (m_log_callback)
      m_log_callback(message);
  }

private:
  const std::string m_name;
  const std::string m_version;

  LogCallback m_log_callback;
  struct Client {
    ReadHandleUP handle;
    MCPTransportUP transport;
    MCPBinderUP binder;
  };
  std::map<MCPTransport *, Client> m_instances;

  llvm::StringMap<std::unique_ptr<Tool>> m_tools;
  std::vector<std::unique_ptr<ResourceProvider>> m_resource_providers;
};

class ServerInfoHandle;

/// Information about this instance of lldb's MCP server for lldb-mcp to use to
/// coordinate connecting an lldb-mcp client.
struct ServerInfo {
  std::string connection_uri;

  /// Writes the server info into a unique file in `~/.lldb`.
  static llvm::Expected<ServerInfoHandle> Write(const ServerInfo &);
  /// Loads any server info saved in `~/.lldb`.
  static llvm::Expected<std::vector<ServerInfo>> Load();
};
llvm::json::Value toJSON(const ServerInfo &);
bool fromJSON(const llvm::json::Value &, ServerInfo &, llvm::json::Path);

/// A handle that tracks the server info on disk and cleans up the disk record
/// once it is no longer referenced.
class ServerInfoHandle {
public:
  explicit ServerInfoHandle(llvm::StringRef filename = "");
  ~ServerInfoHandle();

  ServerInfoHandle(ServerInfoHandle &&other);
  ServerInfoHandle &operator=(ServerInfoHandle &&other) noexcept;

  /// ServerIinfoHandle is not copyable.
  /// @{
  ServerInfoHandle(const ServerInfoHandle &) = delete;
  ServerInfoHandle &operator=(const ServerInfoHandle &) = delete;
  /// @}

  /// Remove the file on disk, if one is tracked.
  void Remove();

private:
  llvm::SmallString<128> m_filename;
};

} // namespace lldb_protocol::mcp

#endif
