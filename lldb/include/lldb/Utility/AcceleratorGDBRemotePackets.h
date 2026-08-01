//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_ACCELERATORGDBREMOTEPACKETS_H
#define LLDB_UTILITY_ACCELERATORGDBREMOTEPACKETS_H

#include "llvm/Support/JSON.h"
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace lldb_private {

struct SymbolValue {
  /// Symbol name as requested in AcceleratorBreakpointInfo::symbol_names.
  std::string name;
  /// Load address of the symbol in the native process, or nullopt if not found.
  std::optional<uint64_t> value;
};

bool fromJSON(const llvm::json::Value &value, SymbolValue &data,
              llvm::json::Path path);
llvm::json::Value toJSON(const SymbolValue &data);

struct AcceleratorBreakpointByName {
  /// Optional shared library name to limit the breakpoint scope.
  std::optional<std::string> shlib;
  /// Function name to set a breakpoint at.
  std::string function_name;
};

bool fromJSON(const llvm::json::Value &value, AcceleratorBreakpointByName &data,
              llvm::json::Path path);
llvm::json::Value toJSON(const AcceleratorBreakpointByName &data);

struct AcceleratorBreakpointByAddress {
  /// Load address in the native debug target.
  uint64_t load_address = 0;
};

bool fromJSON(const llvm::json::Value &value,
              AcceleratorBreakpointByAddress &data, llvm::json::Path path);
llvm::json::Value toJSON(const AcceleratorBreakpointByAddress &data);

/// A breakpoint definition. Clients fill in either \a by_name or
/// \a by_address. If the breakpoint callback needs symbol values from
/// the native process, fill in \a symbol_names — those values will be
/// delivered in the breakpoint hit callback.
struct AcceleratorBreakpointInfo {
  /// Unique breakpoint ID used to identify this breakpoint in the
  /// BreakpointWasHit callback.
  int64_t identifier = 0;
  /// Breakpoint by function name.
  std::optional<AcceleratorBreakpointByName> by_name;
  /// Breakpoint by load address.
  std::optional<AcceleratorBreakpointByAddress> by_address;
  /// Symbol names whose values should be supplied when the breakpoint is hit.
  std::vector<std::string> symbol_names;
};

bool fromJSON(const llvm::json::Value &value, AcceleratorBreakpointInfo &data,
              llvm::json::Path path);
llvm::json::Value toJSON(const AcceleratorBreakpointInfo &data);

/// Sent by the client when a plugin-requested breakpoint is hit.
struct AcceleratorBreakpointHitArgs {
  AcceleratorBreakpointHitArgs() = default;
  AcceleratorBreakpointHitArgs(llvm::StringRef plugin_name)
      : plugin_name(plugin_name) {}

  std::string plugin_name;
  AcceleratorBreakpointInfo breakpoint;
  std::vector<SymbolValue> symbol_values;

  std::optional<uint64_t> GetSymbolValue(llvm::StringRef symbol_name) const;
};

bool fromJSON(const llvm::json::Value &value,
              AcceleratorBreakpointHitArgs &data, llvm::json::Path path);
llvm::json::Value toJSON(const AcceleratorBreakpointHitArgs &data);

/// Information the client needs to connect to an accelerator GDB server. When
/// an AcceleratorActions carries this, the client creates a new target and
/// connects to \a connect_url.
struct AcceleratorConnectionInfo {
  /// Connection URL the client should connect to (as in "process connect
  /// <url>").
  std::string connect_url;
  /// Name of the platform to select when creating the accelerator target. The
  /// platform must be able to handle \a triple and is used to connect to the
  /// accelerator's GDB server.
  std::string platform_name;
  /// Target triple for the accelerator target. Used to ensure the architecture
  /// is compatible with \a platform_name.
  std::string triple;
  /// Path to the executable to use when creating the accelerator target. If
  /// not set, an empty target is created.
  std::optional<std::string> exe_path;
  /// If true, connect synchronously: the client blocks until the accelerator
  /// process is connected and stopped before continuing. If false, the
  /// connection is made asynchronously.
  bool synchronous = false;
};

bool fromJSON(const llvm::json::Value &value, AcceleratorConnectionInfo &data,
              llvm::json::Path path);
llvm::json::Value toJSON(const AcceleratorConnectionInfo &data);

/// Actions to be performed in the native process on behalf of an accelerator
/// plugin. AcceleratorActions are returned in the following contexts:
///
/// - Initialization: in response to the "jAcceleratorPluginInitialize" packet,
///   each plugin returns an AcceleratorActions describing initial breakpoints
///   and other setup needed in the native process.
///
/// - Breakpoint hits: when a native breakpoint requested by a plugin is hit,
///   the AcceleratorBreakpointHitResponse contains an AcceleratorActions that
///   can request additional breakpoints or other actions.
///
/// In future patches, AcceleratorActions will also be returned:
/// - When the native process stops (via NativeProcessIsStopping), allowing
///   plugins to react to arbitrary stop events.
/// - Via accelerator stop reply packets, enabling plugins to inject actions
///   into the native process asynchronously.
struct AcceleratorActions {
  AcceleratorActions() = default;
  AcceleratorActions(llvm::StringRef plugin_name, int64_t action_id)
      : plugin_name(plugin_name), identifier(action_id) {}

  /// Unique name identifying the accelerator plugin.
  std::string plugin_name;
  /// Human-readable label for the accelerator target.
  std::string session_name;
  /// Unique identifier for this action within the plugin.
  int64_t identifier = 0;
  /// New breakpoints to set. Nothing to set if this is empty.
  std::vector<AcceleratorBreakpointInfo> breakpoints;
  /// If set, the client should create a new target and connect to the
  /// accelerator GDB server described here.
  std::optional<AcceleratorConnectionInfo> connect_info;
};

bool fromJSON(const llvm::json::Value &value, AcceleratorActions &data,
              llvm::json::Path path);
llvm::json::Value toJSON(const AcceleratorActions &data);

/// Response from the plugin when a breakpoint is hit.
struct AcceleratorBreakpointHitResponse {
  /// Set to true if this breakpoint should be disabled.
  bool disable_bp = false;
  /// Set to true if the native process should automatically resume after
  /// the breakpoint is hit. When false, the native process will stop and
  /// wait for user interaction.
  bool auto_resume_native = true;
  /// Optional new actions to perform (e.g. set additional breakpoints).
  std::optional<AcceleratorActions> actions;
};

bool fromJSON(const llvm::json::Value &value,
              AcceleratorBreakpointHitResponse &data, llvm::json::Path path);
llvm::json::Value toJSON(const AcceleratorBreakpointHitResponse &data);

} // namespace lldb_private

#endif // LLDB_UTILITY_ACCELERATORGDBREMOTEPACKETS_H
