//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_INTERPRETER_INTERFACES_SCRIPTEDCOMMANDINTERFACE_H
#define LLDB_INTERPRETER_INTERFACES_SCRIPTEDCOMMANDINTERFACE_H

#include "ScriptedInterface.h"
#include "lldb/lldb-private.h"

namespace lldb_private {
class ScriptedCommandInterface : virtual public ScriptedInterface {
public:
  virtual llvm::Expected<StructuredData::GenericSP>
  CreatePluginObject(llvm::StringRef class_name,
                     lldb::DebuggerSP debugger_sp) = 0;

  virtual bool RunRawCommand(llvm::StringRef args,
                             ScriptedCommandSynchronicity synchronicity,
                             CommandReturnObject &cmd_retobj, Status &error,
                             const ExecutionContext &exe_ctx) {
    return false;
  }

  virtual bool RunParsedCommand(Args &args,
                                ScriptedCommandSynchronicity synchronicity,
                                CommandReturnObject &cmd_retobj, Status &error,
                                const ExecutionContext &exe_ctx) {
    return false;
  }

  virtual std::optional<std::string> GetRepeatCommand(Args &args) {
    return std::nullopt;
  }

  virtual StructuredData::DictionarySP
  HandleArgumentCompletion(std::vector<std::string> &args, size_t args_pos,
                           size_t char_in_arg) {
    return {};
  }

  virtual StructuredData::DictionarySP
  HandleOptionArgumentCompletion(llvm::StringRef &long_option,
                                 size_t char_in_arg) {
    return {};
  }

  virtual bool GetShortHelp(std::string &dest) {
    dest.clear();
    return false;
  }

  virtual bool GetLongHelp(std::string &dest) {
    dest.clear();
    return false;
  }

  virtual uint32_t GetFlags() { return 0; }

  virtual StructuredData::ObjectSP GetOptionsDefinition() { return {}; }

  virtual StructuredData::ObjectSP GetArgumentsDefinition() { return {}; }

  virtual void OptionParsingStarted() {}

  virtual bool SetOptionValue(ExecutionContext *exe_ctx,
                              llvm::StringRef long_option,
                              llvm::StringRef value) {
    return false;
  }
};
} // namespace lldb_private

#endif // LLDB_INTERPRETER_INTERFACES_SCRIPTEDCOMMANDINTERFACE_H
