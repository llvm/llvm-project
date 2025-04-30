//===-- GPUGDBRemotePackets.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_GPUGDBREMOTEPACKETS_H
#define LLDB_UTILITY_GPUGDBREMOTEPACKETS_H

#include "lldb/lldb-types.h"
#include "llvm/Support/JSON.h"
#include <string>
#include <vector>

/// See docs/lldb-gdb-remote.txt for more information.
namespace lldb_private {

/// A class that represents a symbol value
struct SymbolValue {
  /// Name of the symbol.
  std::string name;
  /// If the optional doesn't have a value, then the symbol was not available.
  std::optional<uint64_t> value;
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

struct GPUPluginBreakpointHitArgs {
  std::string plugin_name;
  GPUBreakpointInfo breakpoint;
  std::vector<SymbolValue> symbol_values;
};

bool fromJSON(const llvm::json::Value &value, GPUPluginBreakpointHitArgs &data,
              llvm::json::Path path);

llvm::json::Value toJSON(const GPUPluginBreakpointHitArgs &data);

///-----------------------------------------------------------------------------
/// GPUPluginConnectionInfo
///
/// A structure that contains all of the information needed for LLDB to create
/// a reverse connection to a GPU GDB server
///-----------------------------------------------------------------------------
struct GPUPluginConnectionInfo {
  /// A target executable path to use when creating the target.
  std::optional<std::string> exe_path;
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

///-----------------------------------------------------------------------------
/// GPUActions
///
/// A structure that contains action to be taken after a stop or breakpoint hit
/// event.
///-----------------------------------------------------------------------------
struct GPUActions {
  /// The name of the plugin.
  std::string plugin_name;
  /// Optional new breakpoints to set.
  std::optional<std::vector<GPUBreakpointInfo>> breakpoints;
  /// If a GPU connection is available return a connect URL to use to reverse
  /// connect to the GPU GDB server.
  std::optional<GPUPluginConnectionInfo> connect_info;
};

bool fromJSON(const llvm::json::Value &value, 
  GPUActions &data,
  llvm::json::Path path);

llvm::json::Value toJSON(const GPUActions &data);

///-----------------------------------------------------------------------------
/// GPUPluginBreakpointHitResponse
///
/// A response structure from the GPU plugin from hitting a native breakpoint
/// set by the GPU plugin.
///-----------------------------------------------------------------------------
struct GPUPluginBreakpointHitResponse {
  ///< Set to true if this berakpoint should be disabled.
  bool disable_bp = false; 
  /// Optional new breakpoints to set.
  GPUActions actions;
};

bool fromJSON(const llvm::json::Value &value, 
              GPUPluginBreakpointHitResponse &data,
              llvm::json::Path path);

llvm::json::Value toJSON(const GPUPluginBreakpointHitResponse &data);

struct GPUSectionInfo {
  std::string name;
  /// The load address of this section only. If this value is valid, then this
  /// section is loaded at this address, else child sections can be loaded 
  /// individually.
  std::optional<lldb::addr_t> load_address;
  /// Child sections that have individual load addresses can be specified.
  std::vector<GPUSectionInfo> children;
};

bool fromJSON(const llvm::json::Value &value, GPUSectionInfo &data,
              llvm::json::Path path);

llvm::json::Value toJSON(const GPUSectionInfo &data);

struct GPUDynamicLoaderLibraryInfo {
  /// The path to the shared library object file on disk.
  std::string pathname;
  /// The UUID of the shared library if it is known.
  std::optional<std::string> uuid_str;
  /// Set to true if this shared library is being loaded, false if the library
  /// is being unloaded.
  bool load;
  /// The address where the object file is loaded. If this member has a value
  /// the object file is loaded at an address and all sections should be slid to
  /// match this base address. If this member doesn't have a value, then 
  /// individual section's load address must be specified individually if
  /// \a loaded_sections has a value. If this doesn't have a value and the
  /// \a loaded_Section doesn't have a value, this library will be unloaded.
  std::optional<lldb::addr_t> load_address;

  /// If this library is only available as an in memory image of an object file
  /// in the native process, then this address holds the address from which the 
  /// image can be read.
  std::optional<lldb::addr_t> native_memory_address;
  /// If this library is only available as an in memory image of an object file
  /// in the native process, then this size of the in memory image that starts
  /// at \a native_memory_address.
  std::optional<lldb::addr_t> native_memory_size;
  /// If the library exists inside of a file at an offset, \a file_offset will 
  /// have a value that is the offset in bytes from the start of the file 
  /// specified by \a pathname.
  std::optional<uint64_t> file_offset;
  /// If the library exists inside of a file at an offset, \a file_size will 
  /// have a value that indicates the size in bytes of the object file.
  std::optional<uint64_t> file_size;
  /// If the object file specified by this structure has sections that get 
  /// loaded at different times then this will not be empty. If it is empty
  /// the \a load_address must be specified if \a load is true.
  std::vector<GPUSectionInfo> loaded_sections;
};

bool fromJSON(const llvm::json::Value &value, GPUDynamicLoaderLibraryInfo &data,
              llvm::json::Path path);

llvm::json::Value toJSON(const GPUDynamicLoaderLibraryInfo &data);


struct GPUDynamicLoaderArgs {
  /// Set to true to get all shared library information. Set to false to get
  /// only the libraries that were updated since the last call to 
  /// the "jGPUPluginGetDynamicLoaderLibraryInfo" packet.
  bool full;
};

bool fromJSON(const llvm::json::Value &value, GPUDynamicLoaderArgs &data,
              llvm::json::Path path);

llvm::json::Value toJSON(const GPUDynamicLoaderArgs &data);

struct GPUDynamicLoaderResponse {
  /// Set to true to get all shared library information. Set to false to get
  /// only the libraries that were updated since the last call to 
  /// the "jGPUPluginGetDynamicLoaderLibraryInfo" packet.
  std::vector<GPUDynamicLoaderLibraryInfo> library_infos;
};

bool fromJSON(const llvm::json::Value &value, GPUDynamicLoaderResponse &data,
              llvm::json::Path path);

llvm::json::Value toJSON(const GPUDynamicLoaderResponse &data);

} // namespace lldb_private

#endif // LLDB_UTILITY_GPUGDBREMOTEPACKETS_H
