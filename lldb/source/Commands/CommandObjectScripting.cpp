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
#include "lldb/Utility/AnsiTerminal.h"
#include "lldb/Utility/Args.h"

#include "llvm/ADT/StringMap.h"

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
            "scripting extension list [--language <scripting-language> --] "
            "[--json --] [<extension-name> ...]") {
    AddSimpleArgumentList(eArgTypeScriptedExtension, eArgRepeatStar);
  }

  ~CommandObjectScriptingExtensionList() override = default;

  Options *GetOptions() override { return &m_options; }

  void
  HandleArgumentCompletion(CompletionRequest &request,
                           OptionElementVector &opt_element_vector) override {
    lldb_private::CommandCompletions::InvokeCommonCompletionCallbacks(
        GetCommandInterpreter(), lldb::eScriptedExtensionCompletion, request,
        nullptr);
  }

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
      case 'j':
        m_json_format = true;
        break;
      default:
        llvm_unreachable("Unimplemented option");
      }

      return error;
    }

    void OptionParsingStarting(ExecutionContext *execution_context) override {
      m_language = lldb::eScriptLanguageDefault;
      m_json_format = false;
    }

    llvm::ArrayRef<OptionDefinition> GetDefinitions() override {
      return llvm::ArrayRef(g_scripting_extension_list_options);
    }

    lldb::ScriptLanguage m_language = lldb::eScriptLanguageDefault;
    bool m_json_format = false;
  };

protected:
  void DoExecute(Args &command, CommandReturnObject &result) override {
    llvm::StringMap<std::vector<size_t>> grouped_by_extension;
    for (size_t i = 0; i < PluginManager::GetNumScriptedInterfaces(); i++) {
      lldb::ScriptedExtension extension =
          PluginManager::GetScriptedInterfaceExtensionAtIndex(i);
      if (extension == eScriptedExtensionInvalid)
        continue;

      llvm::StringLiteral extension_name =
          ScriptInterpreter::ExtensionToString(extension);
      if (grouped_by_extension.contains(extension_name))
        grouped_by_extension[extension_name].push_back(i);
      else
        grouped_by_extension[extension_name] = {i};
    }

    if (command.GetArgumentCount() > 0) {
      llvm::StringMap<std::vector<size_t>> filtered;
      for (const Args::ArgEntry &arg : command.entries()) {
        lldb::ScriptedExtension extension =
            ScriptInterpreter::StringToExtension(arg.ref());
        if (extension == eScriptedExtensionInvalid) {
          result.AppendErrorWithFormat("no scripted extension named '%s'",
                                       arg.c_str());
          return;
        }
        llvm::StringLiteral extension_name =
            ScriptInterpreter::ExtensionToString(extension);
        auto it = grouped_by_extension.find(extension_name);
        if (it != grouped_by_extension.end())
          filtered[extension_name] = it->second;
      }
      grouped_by_extension = std::move(filtered);
    }

    if (m_options.m_json_format)
      OutputJsonFormat(grouped_by_extension, result);
    else
      OutputTextFormat(grouped_by_extension, result);
  }

private:
  std::vector<std::string>
  GetLanguagesForExtension(const std::vector<size_t> &indices) {
    std::vector<std::string> languages;
    for (const size_t idx : indices) {
      lldb::ScriptLanguage lang =
          PluginManager::GetScriptedInterfaceLanguageAtIndex(idx);
      if (lang != m_options.m_language)
        continue;
      languages.push_back(ScriptInterpreter::LanguageToString(lang));
    }
    return languages;
  }

  void OutputJsonFormat(
      const llvm::StringMap<std::vector<size_t>> &grouped_by_extension,
      CommandReturnObject &result) {
    llvm::json::Array extensions;
    for (const auto &extension_pair : grouped_by_extension) {
      // llvm::json::Value's StringRef constructor does not copy the
      // underlying characters, so every string handed to the JSON structure
      // below must be an owned std::string -- otherwise it dangles by the
      // time the object tree is serialized at the end of this function.
      llvm::json::Array languages;
      for (const std::string &lang :
           GetLanguagesForExtension(extension_pair.second))
        languages.push_back(lang);
      if (languages.empty())
        continue;

      llvm::StringRef desc =
          PluginManager::GetScriptedInterfaceDescriptionAtIndex(
              extension_pair.second[0]);
      ScriptedInterfaceUsages usages =
          PluginManager::GetScriptedInterfaceUsagesAtIndex(
              extension_pair.second[0]);

      llvm::json::Array api_usages;
      for (llvm::StringRef usage : usages.GetSBAPIUsages())
        api_usages.push_back(usage.str());

      llvm::json::Array cmd_usages;
      for (llvm::StringRef usage : usages.GetCommandInterpreterUsages())
        cmd_usages.push_back(usage.str());

      extensions.push_back(llvm::json::Object{
          {"name", extension_pair.first().str()},
          {"description", desc.str()},
          {"languages", std::move(languages)},
          {"api_usages", std::move(api_usages)},
          {"command_interpreter_usages", std::move(cmd_usages)},
      });
    }

    std::string str;
    llvm::raw_string_ostream os(str);
    os << llvm::formatv("{0:2}", llvm::json::Value(std::move(extensions)));
    result.AppendMessage(str);
    result.SetStatus(eReturnStatusSuccessFinishResult);
  }

  void OutputTextFormat(
      const llvm::StringMap<std::vector<size_t>> &grouped_by_extension,
      CommandReturnObject &result) {
    Stream &s = result.GetOutputStream();
    const bool use_color = s.AsRawOstream().colors_enabled();
    auto ansi_code = [use_color](llvm::StringRef code) {
      return ansi::FormatAnsiTerminalCodes(code, use_color);
    };
    const std::string label_color = ansi_code("${ansi.fg.green}${ansi.bold}");
    const std::string name_color = ansi_code("${ansi.fg.cyan}${ansi.bold}");
    const std::string sep_color = ansi_code("${ansi.faint}");
    const std::string reset = ansi_code("${ansi.normal}");
    const std::string separator(
        std::min<uint64_t>(GetDebugger().GetTerminalWidth(), 80), '-');

    s.Printf("Available scripted extension templates:");

    auto print_field = [&](llvm::StringRef key, llvm::StringRef value,
                           const std::string &value_color = std::string()) {
      if (value.empty())
        return;
      s.IndentMore();
      s.Indent();
      s << label_color << key << ": " << reset;
      if (!value_color.empty())
        s << value_color << value << reset;
      else
        s << value;
      s << '\n';
      s.IndentLess();
    };

    size_t num_listed_interface = 0;
    for (const auto &extension_pair : grouped_by_extension) {
      std::vector<std::string> languages =
          GetLanguagesForExtension(extension_pair.second);
      if (languages.empty())
        continue;
      num_listed_interface++;

      s.EOL();
      s << sep_color << separator << reset;
      s.EOL();

      llvm::StringRef desc =
          PluginManager::GetScriptedInterfaceDescriptionAtIndex(
              extension_pair.second[0]);
      ScriptedInterfaceUsages usages =
          PluginManager::GetScriptedInterfaceUsagesAtIndex(
              extension_pair.second[0]);

      print_field("Name", extension_pair.first(), name_color);
      print_field("Description", desc);
      print_field("Language", llvm::join(languages, ""));
      usages.Dump(s, ScriptedInterfaceUsages::UsageKind::API, use_color);
      usages.Dump(s, ScriptedInterfaceUsages::UsageKind::CommandInterpreter,
                  use_color);
    }

    if (!num_listed_interface)
      s << " None\n";

    result.SetStatus(eReturnStatusSuccessFinishResult);
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
