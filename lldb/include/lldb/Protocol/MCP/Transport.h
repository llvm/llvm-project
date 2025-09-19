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
#include "lldb/Protocol/MCP/MCPError.h"
#include "lldb/Protocol/MCP/Protocol.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace lldb_private::transport {
/// Specializations of the JSONTransport protocol functions for MCP.
/// @{
template <>
inline lldb_protocol::mcp::Request
MakeRequest(int64_t id, llvm::StringRef method,
            std::optional<llvm::json::Value> params) {
  return lldb_protocol::mcp::Request{id, method.str(), params};
}
template <>
inline lldb_protocol::mcp::Response
MakeResponse(const lldb_protocol::mcp::Request &req, llvm::Error error) {
  lldb_protocol::mcp::Error protocol_error;
  llvm::handleAllErrors(std::move(error), [&](const llvm::ErrorInfoBase &err) {
    std::error_code cerr = err.convertToErrorCode();
    protocol_error.code = cerr == llvm::inconvertibleErrorCode()
                              ? lldb_protocol::mcp::eErrorCodeInternalError
                              : cerr.value();
    protocol_error.message = err.message();
  });

  return lldb_protocol::mcp::Response{req.id, std::move(protocol_error)};
}
template <>
inline lldb_protocol::mcp::Response
MakeResponse(const lldb_protocol::mcp::Request &req, llvm::json::Value result) {
  return lldb_protocol::mcp::Response{req.id, std::move(result)};
}
template <>
inline lldb_protocol::mcp::Notification
MakeEvent(llvm::StringRef method, std::optional<llvm::json::Value> params) {
  return lldb_protocol::mcp::Notification{method.str(), params};
}
template <>
inline llvm::Expected<llvm::json::Value>
GetResult(const lldb_protocol::mcp::Response &resp) {
  if (const lldb_protocol::mcp::Error *error =
          std::get_if<lldb_protocol::mcp::Error>(&resp.result))
    return llvm::make_error<lldb_protocol::mcp::MCPError>(error->message,
                                                          error->code);
  return std::get<llvm::json::Value>(resp.result);
}
template <> inline int64_t GetId(const lldb_protocol::mcp::Response &resp) {
  return std::get<int64_t>(resp.id);
}
template <>
inline llvm::StringRef GetMethod(const lldb_protocol::mcp::Request &req) {
  return req.method;
}
template <>
inline llvm::StringRef GetMethod(const lldb_protocol::mcp::Notification &evt) {
  return evt.method;
}
template <>
inline llvm::json::Value GetParams(const lldb_protocol::mcp::Request &req) {
  return req.params;
}
template <>
inline llvm::json::Value
GetParams(const lldb_protocol::mcp::Notification &evt) {
  return evt.params;
}
/// @}

} // namespace lldb_private::transport

namespace lldb_protocol::mcp {

/// Generic transport that uses the MCP protocol.
using MCPTransport =
    lldb_private::transport::JSONTransport<int64_t, Request, Response,
                                           Notification>;

/// Generic logging callback, to allow the MCP server / client / transport layer
/// to be independent of the lldb log implementation.
using LogCallback = llvm::unique_function<void(llvm::StringRef message)>;

class Transport final
    : public lldb_private::transport::JSONRPCTransport<int64_t, Request,
                                                       Response, Notification> {
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
