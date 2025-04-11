//===-- GPUGDBRemotePackets.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_GPUGDBREMOTEPACKETS_H
#define LLDB_UTILITY_GPUGDBREMOTEPACKETS_H

#include "llvm/Support/JSON.h"
#include <string>
#include <vector>

/// See docs/lldb-gdb-remote.txt for more information.
namespace lldb_private {

/// A class that represents a symbol value
struct SymbolValue {
  std::string name;
  uint64_t value;
};
bool fromJSON(const llvm::json::Value &value, SymbolValue &data, 
              llvm::json::Path path);

llvm::json::Value toJSON(const SymbolValue &data);

struct GPUBreakpointInfo {
  std::string identifier;
  std::string shlib;
  std::string function_name;
  /// Names of symbols that should be supplied when the breakpoint is hit.
  std::vector<std::string> symbol_names;
};

bool fromJSON(const llvm::json::Value &value, GPUBreakpointInfo &data,
  llvm::json::Path path);

llvm::json::Value toJSON(const GPUBreakpointInfo &data);


struct GPUPluginInfo {
  std::string name;
  std::string description;
  std::vector<GPUBreakpointInfo> breakpoints;
};

bool fromJSON(const llvm::json::Value &value, GPUPluginInfo &data,
  llvm::json::Path path);

llvm::json::Value toJSON(const GPUPluginInfo &data);

struct GPUPluginBreakpointHitArgs {
  std::string plugin_name;
  GPUBreakpointInfo breakpoint;
  std::vector<SymbolValue> symbol_values;
};

bool fromJSON(const llvm::json::Value &value, GPUPluginBreakpointHitArgs &data,
              llvm::json::Path path);

llvm::json::Value toJSON(const GPUPluginBreakpointHitArgs &data);

/// A structure that contains all of the information needed for LLDB to create
/// a reverse connection to a GPU GDB server
struct GPUPluginConnectionInfo {
  /// The name of the platform to select when creating the target.
  std::optional<std::string> platform_name;
  /// The target triple to use as the architecture when creating the target.
  std::optional<std::string> triple;
  /// The connection URL to use with "process connect <url>". 
  std::string connect_url;
};

bool fromJSON(const llvm::json::Value &value, GPUPluginConnectionInfo &data,
              llvm::json::Path path);

llvm::json::Value toJSON(const GPUPluginConnectionInfo &data);

struct GPUPluginBreakpointHitResponse {
  ///< Set to true if this berakpoint should be disabled.
  bool disable_bp = false; 
  /// Optional new breakpoints to set.
  std::optional<std::vector<GPUBreakpointInfo>> breakpoints;
  /// If a GPU connection is available return a connect URL to use to reverse
  /// connect to the GPU GDB server.
  std::optional<std::string> connect_url;
};

bool fromJSON(const llvm::json::Value &value, 
              GPUPluginBreakpointHitResponse &data,
              llvm::json::Path path);

llvm::json::Value toJSON(const GPUPluginBreakpointHitResponse &data);

} // namespace lldb_private

#endif // LLDB_UTILITY_GPUGDBREMOTEPACKETS_H
