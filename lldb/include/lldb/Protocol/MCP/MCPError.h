//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PROTOCOL_MCP_MCPERROR_H
#define LLDB_PROTOCOL_MCP_MCPERROR_H

#include "llvm/Support/Error.h"
#include <string>

namespace lldb_protocol::mcp {

class MCPError : public llvm::ErrorInfo<MCPError> {
public:
  static char ID;

  MCPError(std::string message, int64_t error_code = kInternalError);

  void log(llvm::raw_ostream &OS) const override;
  std::error_code convertToErrorCode() const override;

  const std::string &getMessage() const { return m_message; }

  static constexpr int64_t kResourceNotFound = -32002;
  static constexpr int64_t kInternalError = -32603;

private:
  std::string m_message;
  int m_error_code;
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
