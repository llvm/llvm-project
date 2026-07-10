//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PROTOCOL_MCP_CLIENT_H
#define LLDB_PROTOCOL_MCP_CLIENT_H

#include "lldb/Protocol/MCP/Protocol.h"
#include "lldb/Protocol/MCP/Transport.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/Support/Error.h"
#include <memory>
#include <string>

namespace lldb_protocol::mcp {

/// An MCP client: the counterpart to Server. It speaks the MCP protocol to a
/// remote server over a transport and exposes the protocol's requests as typed,
/// asynchronous calls. Each request completes by invoking its Reply on the
/// transport's MainLoop.
class Client {
public:
  template <typename T> using Reply = lldb_private::transport::Reply<T>;

  Client(std::unique_ptr<MCPTransport> transport,
         LogCallback log_callback = {});
  ~Client() = default;

  Client(const Client &) = delete;
  Client &operator=(const Client &) = delete;

  /// Registers the client with the transport's MainLoop. Must be called before
  /// issuing any request.
  llvm::Error Run();

  /// Sets a handler invoked when the transport disconnects (EOF).
  void SetDisconnectHandler(llvm::unique_function<void()> handler);
  /// Sets a handler invoked on a transport-level error.
  void SetErrorHandler(llvm::unique_function<void(llvm::Error)> handler);

  /// Outgoing MCP requests to the server. Each sends its request and invokes
  /// its Reply asynchronously when the response arrives.
  /// @{
  void Initialize(const InitializeParams &params,
                  Reply<InitializeResult> reply);
  void ToolsList(Reply<ListToolsResult> reply);
  void ToolsCall(const CallToolParams &params, Reply<CallToolResult> reply);
  void ResourcesList(Reply<ListResourcesResult> reply);
  void ResourcesRead(const ReadResourceParams &params,
                     Reply<ReadResourceResult> reply);
  /// @}

  /// MCP notifications.
  void NotifyInitialized();

private:
  void Log(llvm::StringRef message);

  /// Trampolines registered with the binder, which forward to the move-only
  /// handlers below. The binder requires copyable callables, so the handlers
  /// cannot be registered with it directly.
  /// @{
  void HandleDisconnect();
  void HandleError(llvm::Error error);
  /// @}

  std::unique_ptr<MCPTransport> m_transport;
  MCPBinderUP m_binder;
  LogCallback m_log_callback;

  llvm::unique_function<void()> m_disconnect_handler;
  llvm::unique_function<void(llvm::Error)> m_error_handler;

  lldb_private::transport::OutgoingRequest<InitializeResult, InitializeParams>
      m_initialize;
  lldb_private::transport::OutgoingRequest<ListToolsResult, void> m_tools_list;
  lldb_private::transport::OutgoingRequest<CallToolResult, CallToolParams>
      m_tools_call;
  lldb_private::transport::OutgoingRequest<ListResourcesResult, void>
      m_resources_list;
  lldb_private::transport::OutgoingRequest<ReadResourceResult,
                                           ReadResourceParams>
      m_resources_read;
  lldb_private::transport::OutgoingEvent<void> m_notify_initialized;
};

} // namespace lldb_protocol::mcp

#endif
