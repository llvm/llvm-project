//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PROTOCOL_MCP_TOOL_H
#define LLDB_PROTOCOL_MCP_TOOL_H

#include "lldb/Protocol/MCP/Protocol.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include <string>

namespace lldb_protocol::mcp {

class Tool {
public:
  Tool(std::string name, std::string description);
  virtual ~Tool() = default;

  using Reply = llvm::unique_function<void(
      llvm::Expected<lldb_protocol::mcp::ToolsCallResult>)>;

  virtual void Call(const lldb_protocol::mcp::ToolArguments &args,
                    Reply reply) = 0;

  virtual std::optional<llvm::json::Value> GetSchema() const {
    return llvm::json::Object{{"type", "object"}};
  }

  lldb_protocol::mcp::ToolDefinition GetDefinition() const;

  const std::string &GetName() { return m_name; }

private:
  std::string m_name;
  std::string m_description;
};

} // namespace lldb_protocol::mcp

#endif
