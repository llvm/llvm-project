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
#include <sys/types.h>

namespace lldb_protocol::mcp {

struct ProtocolDescriptor {
  using Id = int64_t;
  using Req = Request;
  using Resp = Response;
  using Evt = Notification;

  static inline Id InitialId() { return 0; }
  static inline Request Make(Id id, llvm::StringRef method,
                             std::optional<llvm::json::Value> params) {
    return Request{id, method.str(), params};
  }
  static inline Notification Make(llvm::StringRef method,
                                  std::optional<llvm::json::Value> params) {
    return Notification{method.str(), params};
  }
  static inline Response Make(Req req, llvm::Error error) {
    lldb_protocol::mcp::Error protocol_error;
    llvm::handleAllErrors(
        std::move(error), [&](const llvm::ErrorInfoBase &err) {
          std::error_code cerr = err.convertToErrorCode();
          protocol_error.code =
              cerr == llvm::inconvertibleErrorCode()
                  ? lldb_protocol::mcp::eErrorCodeInternalError
                  : cerr.value();
          protocol_error.message = err.message();
        });

    return Response{req.id, std::move(protocol_error)};
  }
  static inline Response Make(Req req,
                              std::optional<llvm::json::Value> result) {
    return Response{req.id, std::move(result)};
  }
  static inline Id KeyFor(Response r) { return std::get<Id>(r.id); }
  static inline std::string KeyFor(Request r) { return r.method; }
  static inline std::string KeyFor(Notification n) { return n.method; }
  static inline std::optional<llvm::json::Value> Extract(Request r) {
    return r.params;
  }
  static inline llvm::Expected<llvm::json::Value> Extract(Response r) {
    if (const lldb_protocol::mcp::Error *error =
            std::get_if<lldb_protocol::mcp::Error>(&r.result))
      return llvm::make_error<lldb_protocol::mcp::MCPError>(error->message,
                                                            error->code);
    return std::get<llvm::json::Value>(r.result);
  }
  static inline std::optional<llvm::json::Value> Extract(Notification n) {
    return n.params;
  }
};

/// Generic transport that uses the MCP protocol.
using MCPTransport = lldb_private::transport::JSONTransport<ProtocolDescriptor>;
using MCPBinder = lldb_private::transport::Binder<ProtocolDescriptor>;
using MCPBinderUP = std::unique_ptr<MCPBinder>;

/// Generic logging callback, to allow the MCP server / client / transport layer
/// to be independent of the lldb log implementation.
using LogCallback = llvm::unique_function<void(llvm::StringRef message)>;

class Transport final
    : public lldb_private::transport::JSONRPCTransport<ProtocolDescriptor> {
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
