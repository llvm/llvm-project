//===-- ScriptedStopHookInterface.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_INTERPRETER_INTERFACES_SCRIPTEDSTOPHOOKINTERFACE_H
#define LLDB_INTERPRETER_INTERFACES_SCRIPTEDSTOPHOOKINTERFACE_H

#include "lldb/lldb-private.h"

#include "ScriptedInterface.h"

namespace lldb_private {
class ScriptedStopHookInterface : public ScriptedInterface {
public:
  virtual llvm::Expected<StructuredData::GenericSP>
  CreatePluginObject(llvm::StringRef class_name, lldb::TargetSP target_sp,
                     const StructuredDataImpl &args_sp) = 0;

  /// "handle_stop" will return a bool with the meaning "should_stop"...
  /// If nothing is returned, we'll assume we are going to stop.
  /// Also any errors should return true, since we should stop on error.
  virtual llvm::Expected<bool> HandleStop(ExecutionContext &exe_ctx,
                                          lldb::StreamSP output_sp) {
    return true;
  }
};
} // namespace lldb_private

#endif // LLDB_INTERPRETER_INTERFACES_SCRIPTEDSTOPHOOKINTERFACE_H
