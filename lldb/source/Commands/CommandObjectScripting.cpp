//===-- CommandObjectScripting.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CommandObjectScripting.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/DataFormatters/DataVisualization.h"
#include "lldb/Host/Config.h"
#include "lldb/Host/OptionParser.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandOptionArgumentTable.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/Interfaces/ScriptedInterfaceUsages.h"
#include "lldb/Interpreter/OptionArgParser.h"
#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/Utility/Args.h"

using namespace lldb;
using namespace lldb_private;

#define LLDB_OPTIONS_scripting_run
#include "CommandOptions.inc"

class CommandObjectScriptingRun : public CommandObjectRaw {
public:
  CommandObjectScriptingRun(CommandInterpreter &interpreter)
      : CommandObjectRaw(
            interpreter, "scripting run",
            "Invoke the script interpreter with provided code and display any "
            "results.  Start the interactive interpreter if no code is "
            "supplied.",
            "scripting run [--language <scripting-language> --] "
            "[<script-code>]") {}

  ~CommandObjectScriptingRun() override = default;

  Options *GetOptions() override { return &m_options; }

  class CommandOptions : public Options {
  public:
    CommandOptions() = default;
    ~CommandOptions() override = default;
    Status SetOptionValue(uint32_t option_idx, llvm::StringRef option_arg,
                          ExecutionContext *execution_context) override {
      Status error;
      const int short_option = m_getopt_table[option_idx].val;

      switch (short_option) {
      case 'l':
        language = (lldb::ScriptLanguage)OptionArgParser::ToOptionEnum(
            option_arg, GetDefinitions()[option_idx].enum_values,
            eScriptLanguageNone, error);
        if (!error.Success())
          error = Status::FromErrorStringWithFormat(
              "unrecognized value for language '%s'", option_arg.str().c_str());
        break;
      default:
        llvm_unreachable("Unimplemented option");
      }

      return error;
    }

    void OptionParsingStarting(ExecutionContext *execution_context) override {
      language = lldb::eScriptLanguageNone;
    }

    llvm::ArrayRef<OptionDefinition> GetDefinitions() override {
      return llvm::ArrayRef(g_scripting_run_options);
    }

    lldb::ScriptLanguage language = lldb::eScriptLanguageNone;
  };

protected:
  void DoExecute(llvm::StringRef command,
                 CommandReturnObject &result) override {
    // Try parsing the language option but when the command contains a raw part
    // separated by the -- delimiter.
    OptionsWithRaw raw_args(command);
    if (raw_args.HasArgs()) {
      if (!ParseOptions(raw_args.GetArgs(), result))
        return;
      command = raw_args.GetRawPart();
    }

    lldb::ScriptLanguage language =
        (m_options.language == lldb::eScriptLanguageNone)
            ? m_interpreter.GetDebugger().GetScriptLanguage()
            : m_options.language;

    if (language == lldb::eScriptLanguageNone) {
      result.AppendError(
          "the script-lang setting is set to none - scripting not available");
      return;
    }

    ScriptInterpreter *script_interpreter =
        GetDebugger().GetScriptInterpreter(true, language);

    if (script_interpreter == nullptr) {
      result.AppendError("no script interpreter");
      return;
    }

    // Script might change Python code we use for formatting. Make sure we keep
    // up to date with it.
    DataVisualization::ForceUpdate();

    if (command.empty()) {
      script_interpreter->ExecuteInterpreterLoop();
      result.SetStatus(eReturnStatusSuccessFinishNoResult);
      return;
    }

    // We can do better when reporting the status of one-liner script execution.
    if (script_interpreter->ExecuteOneLine(command, &result))
      result.SetStatus(eReturnStatusSuccessFinishNoResult);
    else
      result.SetStatus(eReturnStatusFailed);
  }

private:
  CommandOptions m_options;
};

#define LLDB_OPTIONS_scripting_extension_list
#include "CommandOptions.inc"

class CommandObjectScriptingExtensionList : public CommandObjectParsed {
public:
  CommandObjectScriptingExtensionList(CommandInterpreter &interpreter)
      : CommandObjectParsed(
            interpreter, "scripting extension list",
            "List all the available scripting extension templates. ",
            "scripting template list [--language <scripting-language> --]") {}

  ~CommandObjectScriptingExtensionList() override = default;

  Options *GetOptions() override { return &m_options; }

  class CommandOptions : public Options {
  public:
    CommandOptions() = default;
    ~CommandOptions() override = default;
    Status SetOptionValue(uint32_t option_idx, llvm::StringRef option_arg,
                          ExecutionContext *execution_context) override {
      Status error;
      const int short_option = m_getopt_table[option_idx].val;

      switch (short_option) {
      case 'l':
        m_language = (lldb::ScriptLanguage)OptionArgParser::ToOptionEnum(
            option_arg, GetDefinitions()[option_idx].enum_values,
            eScriptLanguageNone, error);
        if (!error.Success())
          error = Status::FromErrorStringWithFormatv(
              "unrecognized value for language '{0}'", option_arg);
        break;
      default:
        llvm_unreachable("Unimplemented option");
      }

      return error;
    }

    void OptionParsingStarting(ExecutionContext *execution_context) override {
      m_language = lldb::eScriptLanguageDefault;
    }

    llvm::ArrayRef<OptionDefinition> GetDefinitions() override {
      return llvm::ArrayRef(g_scripting_extension_list_options);
    }

    lldb::ScriptLanguage m_language = lldb::eScriptLanguageDefault;
  };

protected:
  void DoExecute(Args &command, CommandReturnObject &result) override {
    Stream &s = result.GetOutputStream();
    s.Printf("Available scripted extension templates:");

    auto print_field = [&s](llvm::StringRef key, llvm::StringRef value) {
      if (!value.empty()) {
        s.IndentMore();
        s.Indent();
        s << key << ": " << value << '\n';
        s.IndentLess();
      }
    };

    size_t num_listed_interface = 0;
    size_t num_extensions = PluginManager::GetNumScriptedInterfaces();
    for (size_t i = 0; i < num_extensions; i++) {
      llvm::StringRef plugin_name =
          PluginManager::GetScriptedInterfaceNameAtIndex(i);
      if (plugin_name.empty())
        break;

      lldb::ScriptLanguage lang =
          PluginManager::GetScriptedInterfaceLanguageAtIndex(i);
      if (lang != m_options.m_language)
        continue;

      if (!num_listed_interface)
        s.EOL();

      num_listed_interface++;

      llvm::StringRef desc =
          PluginManager::GetScriptedInterfaceDescriptionAtIndex(i);
      ScriptedInterfaceUsages usages =
          PluginManager::GetScriptedInterfaceUsagesAtIndex(i);

      print_field("Name", plugin_name);
      print_field("Language", ScriptInterpreter::LanguageToString(lang));
      print_field("Description", desc);
      usages.Dump(s, ScriptedInterfaceUsages::UsageKind::API);
      usages.Dump(s, ScriptedInterfaceUsages::UsageKind::CommandInterpreter);

      if (i != num_extensions - 1)
        s.EOL();
    }

    if (!num_listed_interface)
      s << " None\n";
  }

private:
  CommandOptions m_options;
};

class CommandObjectMultiwordScriptingExtension : public CommandObjectMultiword {
public:
  CommandObjectMultiwordScriptingExtension(CommandInterpreter &interpreter)
      : CommandObjectMultiword(
            interpreter, "scripting extension",
            "Commands for operating on the scripting extensions.",
            "scripting extension [<subcommand-options>]") {
    LoadSubCommand(
        "list",
        CommandObjectSP(new CommandObjectScriptingExtensionList(interpreter)));
  }

  ~CommandObjectMultiwordScriptingExtension() override = default;
};

CommandObjectMultiwordScripting::CommandObjectMultiwordScripting(
    CommandInterpreter &interpreter)
    : CommandObjectMultiword(
          interpreter, "scripting",
          "Commands for operating on the scripting functionalities.",
          "scripting <subcommand> [<subcommand-options>]") {
  LoadSubCommand("run",
                 CommandObjectSP(new CommandObjectScriptingRun(interpreter)));
  LoadSubCommand("extension",
                 CommandObjectSP(new CommandObjectMultiwordScriptingExtension(
                     interpreter)));
}

CommandObjectMultiwordScripting::~CommandObjectMultiwordScripting() = default;
