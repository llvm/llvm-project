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

#include "ScriptedInterface.h"

namespace lldb_private {
class ScriptedHookInterface : public ScriptedInterface {
public:
  /// Describes which hook callback methods the Python class implements.
  struct SupportedHookMethods {
    bool handle_module_loaded = false;
    bool handle_module_unloaded = false;
    bool handle_stop = false;

    bool any() const {
      return handle_module_loaded || handle_module_unloaded || handle_stop;
    }
  };

  virtual llvm::Expected<StructuredData::GenericSP>
  CreatePluginObject(llvm::StringRef class_name, lldb::TargetSP target_sp,
                     const StructuredDataImpl &args_sp) = 0;

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
};
} // namespace lldb_private

#endif // LLDB_INTERPRETER_INTERFACES_SCRIPTEDHOOKINTERFACE_H
