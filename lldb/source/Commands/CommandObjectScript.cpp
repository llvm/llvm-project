//===-- CommandObjectScript.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CommandObjectScript.h"
#include "lldb/Core/Debugger.h"
#include "lldb/DataFormatters/DataVisualization.h"
#include "lldb/Host/Config.h"
#include "lldb/Host/OptionParser.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/OptionArgParser.h"
#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/Utility/Args.h"

using namespace lldb;
using namespace lldb_private;

static constexpr OptionEnumValueElement g_script_option_enumeration[] = {
    {
        eScriptLanguagePython,
        "python",
        "Python",
    },
    {
        eScriptLanguageLua,
        "lua",
        "Lua",
    },
    {
        eScriptLanguageNone,
        "default",
        "The default scripting language.",
    },
};

static constexpr OptionEnumValues ScriptOptionEnum() {
  return OptionEnumValues(g_script_option_enumeration);
}

#define LLDB_OPTIONS_script
#include "CommandOptions.inc"

Status CommandObjectScript::CommandOptions::SetOptionValue(
    uint32_t option_idx, llvm::StringRef option_arg,
    ExecutionContext *execution_context) {
  Status error;
  const int short_option = m_getopt_table[option_idx].val;

  switch (short_option) {
  case 'l':
    language = (lldb::ScriptLanguage)OptionArgParser::ToOptionEnum(
        option_arg, GetDefinitions()[option_idx].enum_values,
        eScriptLanguageNone, error);
    if (!error.Success())
      error.SetErrorStringWithFormat("unrecognized value for language '%s'",
                                     option_arg.str().c_str());
    break;
  default:
    llvm_unreachable("Unimplemented option");
  }

  return error;
}

void CommandObjectScript::CommandOptions::OptionParsingStarting(
    ExecutionContext *execution_context) {
  language = lldb::eScriptLanguageNone;
}

llvm::ArrayRef<OptionDefinition>
CommandObjectScript::CommandOptions::GetDefinitions() {
  return llvm::makeArrayRef(g_script_options);
}

CommandObjectScript::CommandObjectScript(CommandInterpreter &interpreter)
    : CommandObjectRaw(
          interpreter, "script",
          "Invoke the script interpreter with provided code and display any "
          "results.  Start the interactive interpreter if no code is supplied.",
          "script [--language <scripting-language> --] [<script-code>]") {}

CommandObjectScript::~CommandObjectScript() = default;

bool CommandObjectScript::DoExecute(llvm::StringRef command,
                                    CommandReturnObject &result) {
  // Try parsing the language option but when the command contains a raw part
  // separated by the -- delimiter.
  OptionsWithRaw raw_args(command);
  if (raw_args.HasArgs()) {
    if (!ParseOptions(raw_args.GetArgs(), result))
      return false;
    command = raw_args.GetRawPart();
  }

  lldb::ScriptLanguage language =
      (m_options.language == lldb::eScriptLanguageNone)
          ? m_interpreter.GetDebugger().GetScriptLanguage()
          : m_options.language;

  if (language == lldb::eScriptLanguageNone) {
    result.AppendError(
        "the script-lang setting is set to none - scripting not available");
    return false;
  }

  ScriptInterpreter *script_interpreter =
      GetDebugger().GetScriptInterpreter(true, language);

  if (script_interpreter == nullptr) {
    result.AppendError("no script interpreter");
    return false;
  }

  // Script might change Python code we use for formatting. Make sure we keep
  // up to date with it.
  DataVisualization::ForceUpdate();

  if (command.empty()) {
    script_interpreter->ExecuteInterpreterLoop();
    result.SetStatus(eReturnStatusSuccessFinishNoResult);
    return result.Succeeded();
  }

  // We can do better when reporting the status of one-liner script execution.
  if (script_interpreter->ExecuteOneLine(command, &result))
    result.SetStatus(eReturnStatusSuccessFinishNoResult);
  else
    result.SetStatus(eReturnStatusFailed);

  return result.Succeeded();
}
