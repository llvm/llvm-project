//===-- ScriptedModuleHookInterface.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_INTERPRETER_INTERFACES_SCRIPTEDMODULEHOOKINTERFACE_H
#define LLDB_INTERPRETER_INTERFACES_SCRIPTEDMODULEHOOKINTERFACE_H

#include "lldb/lldb-private.h"

#include "ScriptedInterface.h"

namespace lldb_private {
class ScriptedModuleHookInterface : public ScriptedInterface {
public:
  virtual llvm::Expected<StructuredData::GenericSP>
  CreatePluginObject(llvm::StringRef class_name, lldb::TargetSP target_sp,
                     const StructuredDataImpl &args_sp) = 0;

  /// Called when modules are loaded into the target. Unlike stop hooks,
  /// module hooks do not control whether the process should stop.
  virtual void HandleModuleLoaded(lldb::StreamSP &output_sp) {}

  /// Called when modules are unloaded from the target. Optional for
  /// scripted hooks; if not implemented, the hook silently does nothing
  /// on unload.
  virtual void HandleModuleUnloaded(lldb::StreamSP &output_sp) {}
};
} // namespace lldb_private

#endif // LLDB_INTERPRETER_INTERFACES_SCRIPTEDMODULEHOOKINTERFACE_H
