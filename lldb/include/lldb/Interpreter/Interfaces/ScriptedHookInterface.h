//===-- ScriptedHookInterface.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_INTERPRETER_INTERFACES_SCRIPTEDHOOKINTERFACE_H
#define LLDB_INTERPRETER_INTERFACES_SCRIPTEDHOOKINTERFACE_H

#include "lldb/lldb-private.h"
#include "lldb/Core/Address.h"
#include "ScriptedInterface.h"

namespace lldb_private {
class ScriptedHookInterface : public ScriptedInterface {
public:
  /// Describes which hook callback methods the Python class implements.
  struct SupportedHookMethods {
    bool handle_module_loaded = false;
    bool handle_module_unloaded = false;
    bool handle_stop = false;
    bool handle_resolve_addr = false;

    bool any() const {
      return handle_module_loaded || handle_module_unloaded || handle_stop || handle_resolve_addr;
    }
  };

  virtual llvm::Expected<StructuredData::GenericSP>
  CreatePluginObject(const ScriptedMetadata &scripted_metadata,
                     lldb::TargetSP target_sp) = 0;

  /// Check which hook callback methods the Python class implements.
  /// Called after CreatePluginObject to determine the trigger mask.
  virtual SupportedHookMethods GetSupportedMethods() { return {}; }

  /// Called when modules are loaded into the target.
  virtual void HandleModuleLoaded(lldb::StreamSP &output_sp) {}

  /// Called when modules are unloaded from the target. Optional.
  virtual void HandleModuleUnloaded(lldb::StreamSP &output_sp) {}

  /// Called when the process stops. Returns "should_stop" if false, the
  /// process will continue. Defaults to true (stop on unimplemented).
  virtual llvm::Expected<bool> HandleStop(ExecutionContext &exe_ctx,
                                          lldb::StreamSP &output_sp) {
    return true;
  }

  /// Called when the target tried to resolve an address but wasn't able to 
  /// resolve it to an object file section. This allows plug-ins to resolve
  /// an address on demand. JIT plug-ins can be completely implemented using
  /// ScriptedHookInterface plug-ins and can lazily load the JIT'ed information
  /// as needed instead of setting breakpoints
  ///
  /// \param[in] load_addr
  ///   The load address to resolve.
  ///
  /// \param[out] addr
  ///   The section offset address that was resolved if \a true is returned.
  ///
  /// \param[in] output_sp
  ///   An output stream to use for logging.
  ///
  /// \return
  ///   True if the address was resolved, false otherwise.
  virtual std::optional<Address>
  HandleResolveAddress(lldb::addr_t load_addr, lldb::StreamSP &output_sp) {
    return std::nullopt;
  }
};
} // namespace lldb_private

#endif // LLDB_INTERPRETER_INTERFACES_SCRIPTEDHOOKINTERFACE_H
