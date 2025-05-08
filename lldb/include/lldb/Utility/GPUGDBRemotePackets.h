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

///-----------------------------------------------------------------------------
/// GPUBreakpointByName
///
/// A structure that contains information on how to set a breakpoint by function
/// name with optional shared library name.
///-----------------------------------------------------------------------------

struct GPUBreakpointByName {
  /// An optional breakpoint shared library name to limit the scope of the
  /// breakpoint to a specific shared library.
  std::optional<std::string> shlib;
  /// The name of the function to set a breakpoint at.
  std::string function_name;
};

bool fromJSON(const llvm::json::Value &value, GPUBreakpointByName &data,
              llvm::json::Path path);

llvm::json::Value toJSON(const GPUBreakpointByName &data);

///-----------------------------------------------------------------------------
/// GPUBreakpointByAddress
///
/// A structure that contains information on how to set a breakpoint by address.
///-----------------------------------------------------------------------------
struct GPUBreakpointByAddress {
  /// A valid load address in the current native debug target.
  lldb::addr_t load_address;
};

bool fromJSON(const llvm::json::Value &value, GPUBreakpointByAddress &data,
              llvm::json::Path path);

llvm::json::Value toJSON(const GPUBreakpointByAddress &data);

///-----------------------------------------------------------------------------
/// GPUBreakpointInfo
///
/// A breakpoint definition structure.
///
/// Clients should either fill in the \a name_info or the \a addr_info. If the
/// breakpoint callback needs some symbols from the native process, they can
/// fill in the array of symbol names with any symbol names that are needed. 
/// These symbol values will be delivered in the breakpoint callback to the GPU
/// plug-in.
///-----------------------------------------------------------------------------
struct GPUBreakpointInfo {
  std::string identifier;
  /// An optional breakpoint by name info.
  std::optional<GPUBreakpointByName> name_info;
  /// An optional load address to set a breakpoint at in the native process.
  std::optional<GPUBreakpointByAddress> addr_info;
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

  std::optional<uint64_t> GetSymbolValue(llvm::StringRef symbol_name);
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
/// A structure used by the native process that is debugging the GPU that
/// contains actions to be performed after:
///
/// - GPU Initilization in response to the "jGPUPluginInitialize" packet sent to
///   the native process' lldb-server that contains GPU plugins. This packet is
///   sent to the ProcessGDBRemote for the native process one time when a native
///   process is being attached or launched.
///
/// - When a native breakpoint that was requested by the GPU plugin is hit, the
///   native process in LLDB will call into the native process' GDB server and
///   have it call the GPU plug-in method:
///
///     GPUPluginBreakpointHitResponse 
///     LLDBServerPlugin::BreakpointWasHit(GPUPluginBreakpointHitArgs &args);
///
///   The GPUPluginBreakpointHitResponse contains a GPUActions member that will
///   be encoded and sent back to the ProcessGDBRemote for the native process. 
///
/// - Anytime the native process stops, the native process' GDB server will ask
///   each GPU plug-in if there are any actions it would like to report, the
///   native process' lldb-server will call the GPU plug-in method:
///
///     std::optional<GPUActions> LLDBServerPlugin::NativeProcessIsStopping();
///
///   If GPUActions are returned from this method, they will be encoded into the
///   native process' stop reply packet and handled in ProcessGDBRemote for the
///   native process.
///-----------------------------------------------------------------------------
struct GPUActions {
  /// The name of the plugin.
  std::string plugin_name;
  /// New breakpoints to set. Nothing to set if this is empty.
  std::vector<GPUBreakpointInfo> breakpoints;
  /// If a GPU connection is available return a connect URL to use to reverse
  /// connect to the GPU GDB server as a separate process.
  std::optional<GPUPluginConnectionInfo> connect_info;
  /// Set this to true if the native plug-in should tell the ProcessGDBRemote
  /// in LLDB for the GPU process to load libraries. This allows the native 
  /// process to be notified that it should query for the shared libraries on 
  /// the GPU connection.
  bool load_libraries = false;
  /// Set this to true if the native plug-in resume the GPU process.
  bool resume_gpu_process = false;
};

bool fromJSON(const llvm::json::Value &value, 
  GPUActions &data,
  llvm::json::Path path);

llvm::json::Value toJSON(const GPUActions &data);


struct GPUSectionInfo {
  /// Name of the section to load. If there are multiple sections, each section
  /// will be looked up and then a child section within the previous section
  /// will be looked up. This allows plug-ins to specify a hiearchy of sections
  /// in the case where section names are not unique. A valid example looks 
  /// like: ["PT_LOAD[0]", ".text"]. If there is only one section name, LLDB
  /// will find the first section that matches that name.
  std::vector<std::string> names;
  /// The load address of this section only. If this value is valid, then this
  /// section is loaded at this address, else child sections can be loaded 
  /// individually.
  lldb::addr_t load_address;
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

  /// If the object file specified by this structure has sections that get 
  /// loaded at different times then this will not be empty. If it is empty
  /// the \a load_address must be specified if \a load is true.
  std::vector<GPUSectionInfo> loaded_sections;

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
};

bool fromJSON(const llvm::json::Value &value, GPUDynamicLoaderLibraryInfo &data,
              llvm::json::Path path);

llvm::json::Value toJSON(const GPUDynamicLoaderLibraryInfo &data);



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
