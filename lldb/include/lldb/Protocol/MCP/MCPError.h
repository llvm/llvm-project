//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PROTOCOL_MCP_MCPERROR_H
#define LLDB_PROTOCOL_MCP_MCPERROR_H

#include "lldb/Protocol/MCP/Protocol.h"
#include "llvm/Support/Error.h"
#include <string>

namespace lldb_protocol::mcp {

class MCPError : public llvm::ErrorInfo<MCPError> {
public:
  static char ID;

  MCPError(std::string message, int64_t error_code = eErrorCodeInternalError);

  void log(llvm::raw_ostream &OS) const override;
  std::error_code convertToErrorCode() const override;

  const std::string &getMessage() const { return m_message; }

  lldb_protocol::mcp::Error toProtocolError() const;

private:
  std::string m_message;
  int64_t m_error_code;
};

class UnsupportedURI : public llvm::ErrorInfo<UnsupportedURI> {
public:
  static char ID;

  UnsupportedURI(std::string uri);

  void log(llvm::raw_ostream &OS) const override;
  std::error_code convertToErrorCode() const override;

private:
  std::string m_uri;
};

} // namespace lldb_protocol::mcp

#endif
