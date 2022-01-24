//===-- CommandObjectScript.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_INTERPRETER_COMMANDOBJECTSCRIPT_H
#define LLDB_SOURCE_INTERPRETER_COMMANDOBJECTSCRIPT_H

#include "lldb/Interpreter/CommandObject.h"

namespace lldb_private {

class CommandObjectScript : public CommandObjectRaw {
public:
  CommandObjectScript(CommandInterpreter &interpreter);
  ~CommandObjectScript() override;
  Options *GetOptions() override { return &m_options; }

  class CommandOptions : public Options {
  public:
    CommandOptions() {}
    ~CommandOptions() override = default;
    Status SetOptionValue(uint32_t option_idx, llvm::StringRef option_arg,
                          ExecutionContext *execution_context) override;
    void OptionParsingStarting(ExecutionContext *execution_context) override;
    llvm::ArrayRef<OptionDefinition> GetDefinitions() override;
    lldb::ScriptLanguage language = lldb::eScriptLanguageNone;
  };

protected:
  bool DoExecute(llvm::StringRef command, CommandReturnObject &result) override;

private:
  CommandOptions m_options;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_INTERPRETER_COMMANDOBJECTSCRIPT_H
