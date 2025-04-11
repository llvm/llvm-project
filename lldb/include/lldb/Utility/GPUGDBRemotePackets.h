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
bool fromJSON(const llvm::json::Value &value, SymbolValue &info, 
              llvm::json::Path path);

llvm::json::Value toJSON(const SymbolValue &packet);

struct GPUBreakpointInfo {
  std::string identifier;
  std::string shlib;
  std::string function_name;
  /// Names of symbols that should be supplied when the breakpoint is hit.
  std::vector<std::string> symbol_names;
};

bool fromJSON(const llvm::json::Value &value, GPUBreakpointInfo &info,
  llvm::json::Path path);

llvm::json::Value toJSON(const GPUBreakpointInfo &packet);


struct GPUPluginInfo {
  std::string name;
  std::string description;
  std::vector<GPUBreakpointInfo> breakpoints;
};

bool fromJSON(const llvm::json::Value &value, GPUPluginInfo &info,
  llvm::json::Path path);

llvm::json::Value toJSON(const GPUPluginInfo &packet);

struct GPUPluginBreakpointHitArgs {
  std::string plugin_name;
  GPUBreakpointInfo breakpoint;
  std::vector<SymbolValue> symbol_values;
};

bool fromJSON(const llvm::json::Value &value, GPUPluginBreakpointHitArgs &info,
              llvm::json::Path path);

llvm::json::Value toJSON(const GPUPluginBreakpointHitArgs &packet);

} // namespace lldb_private

#endif // LLDB_UTILITY_GPUGDBREMOTEPACKETS_H
