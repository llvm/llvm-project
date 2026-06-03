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

/// A section load address entry for dynamic loader library info.
struct AcceleratorSectionInfo {
  /// Hierarchical section names. If there are multiple names, each successive
  /// name is looked up as a child section of the previous one. Example:
  /// ["PT_LOAD[0]", ".text"].
  std::vector<std::string> names;
  /// The load address of this section.
  uint64_t load_address = 0;
};

bool fromJSON(const llvm::json::Value &value, AcceleratorSectionInfo &data,
              llvm::json::Path path);
llvm::json::Value toJSON(const AcceleratorSectionInfo &data);

/// Information about a shared library for the accelerator dynamic loader.
struct AcceleratorDynamicLoaderLibraryInfo {
  /// Path to the shared library object file on disk.
  std::string pathname;
  /// UUID of the shared library, if known.
  std::optional<std::string> uuid_str;
  /// True if loading, false if unloading.
  bool load = true;
  /// Base load address for the entire object file. If set, all sections are
  /// slid to match. If not set, use \a loaded_sections or file addresses.
  std::optional<uint64_t> load_address;
  /// Per-section load addresses for object files with sections loaded at
  /// different addresses.
  std::vector<AcceleratorSectionInfo> loaded_sections;
  /// Address in the native process where the object file image can be read.
  std::optional<uint64_t> native_memory_address;
  /// Size of the in-memory image starting at \a native_memory_address.
  std::optional<uint64_t> native_memory_size;
  /// Byte offset within the file specified by \a pathname.
  std::optional<uint64_t> file_offset;
  /// Size in bytes of the object file within the containing file.
  std::optional<uint64_t> file_size;
};

bool fromJSON(const llvm::json::Value &value,
              AcceleratorDynamicLoaderLibraryInfo &data,
              llvm::json::Path path);
llvm::json::Value toJSON(const AcceleratorDynamicLoaderLibraryInfo &data);

/// Arguments for the jAcceleratorPluginGetDynamicLoaderLibraryInfo packet.
struct AcceleratorDynamicLoaderArgs {
  /// Name of the accelerator plugin to query.
  std::string plugin_name;
  /// If true, return all libraries. If false, return only updates since the
  /// last query.
  bool full = true;
};

bool fromJSON(const llvm::json::Value &value,
              AcceleratorDynamicLoaderArgs &data, llvm::json::Path path);
llvm::json::Value toJSON(const AcceleratorDynamicLoaderArgs &data);

/// Response from the jAcceleratorPluginGetDynamicLoaderLibraryInfo packet.
struct AcceleratorDynamicLoaderResponse {
  std::vector<AcceleratorDynamicLoaderLibraryInfo> library_infos;
};

bool fromJSON(const llvm::json::Value &value,
              AcceleratorDynamicLoaderResponse &data, llvm::json::Path path);
llvm::json::Value toJSON(const AcceleratorDynamicLoaderResponse &data);

} // namespace lldb_private

#endif // LLDB_UTILITY_ACCELERATORGDBREMOTEPACKETS_H
