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
#include "lldb/Interpreter/OptionArgParser.h"
#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/Utility/Args.h"

using namespace lldb;
using namespace lldb_private;

#pragma mark CommandObjectScriptingRun

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
          error.SetErrorStringWithFormat("unrecognized value for language '%s'",
                                         option_arg.str().c_str());
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

#pragma mark CommandObjectScriptingTemplateList

#define LLDB_OPTIONS_scripting_template_list
#include "CommandOptions.inc"

class CommandObjectScriptingTemplateList : public CommandObjectParsed {
public:
  CommandObjectScriptingTemplateList(CommandInterpreter &interpreter)
      : CommandObjectParsed(
            interpreter, "scripting template list",
            "List all the available scripting affordances templates. ",
            "scripting template list [--language <scripting-language> --]") {}

  ~CommandObjectScriptingTemplateList() override = default;

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
          error.SetErrorStringWithFormat("unrecognized value for language '%s'",
                                         option_arg.str().c_str());
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
      return llvm::ArrayRef(g_scripting_template_list_options);
    }

    lldb::ScriptLanguage language = lldb::eScriptLanguageNone;
  };

protected:
  void DoExecute(Args &command, CommandReturnObject &result) override {
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

    Stream &s = result.GetOutputStream();
    s.Printf("Available scripted affordances:\n");

    auto print_field = [&s](llvm::StringRef key, llvm::StringRef value,
                            bool check_validy = false) {
      if (!check_validy || !value.empty()) {
        s.IndentMore();
        s.Indent();
        s << key << ": " << value << '\n';
        s.IndentLess();
      }
    };
    auto print_usages = [&s](llvm::StringRef usage_kind,
                             std::vector<llvm::StringRef> &usages) {
      s.IndentMore();
      s.Indent();
      s << usage_kind << " Usages:";
      if (usages.empty())
        s << " No usages.\n";
      else if (usages.size() == 1)
        s << " " << usages.front() << '\n';
      else {
        s << '\n';
        for (llvm::StringRef usage : usages) {
          s.IndentMore();
          s.Indent();
          s << usage << '\n';
          s.IndentLess();
        }
      }
      s.IndentLess();
    };

    size_t i = 0;
    for (llvm::StringRef plugin_name =
             PluginManager::GetScriptedInterfaceNameAtIndex(i);
         !plugin_name.empty();) {

      llvm::StringRef desc =
          PluginManager::GetScriptedInterfaceDescriptionAtIndex(i);
      lldb::ScriptLanguage lang =
          PluginManager::GetScriptedInterfaceLanguageAtIndex(i);
      std::vector<llvm::StringRef> ci_usages =
          PluginManager::GetScriptedInterfaceCommandInterpreterUsagesAtIndex(i);
      std::vector<llvm::StringRef> api_usages =
          PluginManager::GetScriptedInterfaceAPIUsagesAtIndex(i);

      print_field("Name", plugin_name);
      switch (lang) {
      case eScriptLanguagePython:
        print_field("Language", "Python");
        break;
      case eScriptLanguageLua:
        print_field("Language", "Lua");
        break;
      default:
        break;
      }
      print_field("Description", desc);
      print_usages("Command Interpreter", ci_usages);
      print_usages("API", api_usages);

      plugin_name = PluginManager::GetScriptedInterfaceNameAtIndex(++i);
      if (!plugin_name.empty())
        s.EOL();
    }
  }

private:
  CommandOptions m_options;
};

#pragma mark CommandObjectMultiwordScriptingTemplate

// CommandObjectMultiwordScriptingTemplate

class CommandObjectMultiwordScriptingTemplate : public CommandObjectMultiword {
public:
  CommandObjectMultiwordScriptingTemplate(CommandInterpreter &interpreter)
      : CommandObjectMultiword(
            interpreter, "scripting template",
            "Commands for operating on the scripting templates.",
            "scripting template [<subcommand-options>]") {
    LoadSubCommand(
        "list",
        CommandObjectSP(new CommandObjectScriptingTemplateList(interpreter)));
  }

  ~CommandObjectMultiwordScriptingTemplate() override = default;
};

#pragma mark CommandObjectMultiwordScripting

// CommandObjectMultiwordScripting

CommandObjectMultiwordScripting::CommandObjectMultiwordScripting(
    CommandInterpreter &interpreter)
    : CommandObjectMultiword(
          interpreter, "scripting",
          "Commands for operating on the scripting functionnalities.",
          "scripting <subcommand> [<subcommand-options>]") {
  LoadSubCommand("run",
                 CommandObjectSP(new CommandObjectScriptingRun(interpreter)));
  LoadSubCommand("template",
                 CommandObjectSP(
                     new CommandObjectMultiwordScriptingTemplate(interpreter)));
}

CommandObjectMultiwordScripting::~CommandObjectMultiwordScripting() = default;
