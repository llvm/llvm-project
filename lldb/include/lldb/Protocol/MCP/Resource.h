//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PROTOCOL_MCP_RESOURCE_H
#define LLDB_PROTOCOL_MCP_RESOURCE_H

#include "lldb/Protocol/MCP/Protocol.h"
#include <vector>

namespace lldb_protocol::mcp {

class ResourceProvider {
public:
  ResourceProvider() = default;
  virtual ~ResourceProvider() = default;

  virtual std::vector<lldb_protocol::mcp::Resource> GetResources() const = 0;
  virtual llvm::Expected<lldb_protocol::mcp::ReadResourceResult>
  ReadResource(llvm::StringRef uri) const = 0;
};

} // namespace lldb_protocol::mcp

#endif
