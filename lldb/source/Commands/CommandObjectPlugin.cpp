//===-- CommandObjectPlugin.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CommandObjectPlugin.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/OptionParser.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"

using namespace lldb;
using namespace lldb_private;

class CommandObjectPluginLoad : public CommandObjectParsed {
public:
  CommandObjectPluginLoad(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "plugin load",
                            "Import a dylib that implements an LLDB plugin.",
                            nullptr) {
    AddSimpleArgumentList(eArgTypeFilename);
  }

  ~CommandObjectPluginLoad() override = default;

protected:
  void DoExecute(Args &command, CommandReturnObject &result) override {
    size_t argc = command.GetArgumentCount();

    if (argc != 1) {
      result.AppendError("'plugin load' requires one argument");
      return;
    }

    Status error;

    FileSpec dylib_fspec(command[0].ref());
    FileSystem::Instance().Resolve(dylib_fspec);

    if (GetDebugger().LoadPlugin(dylib_fspec, error))
      result.SetStatus(eReturnStatusSuccessFinishResult);
    else {
      result.AppendError(error.AsCString());
    }
  }
};

namespace {
// Helper function to perform an action on each matching plugin.
// The action callback is given the containing namespace along with plugin info
// for each matching plugin.
static int ActOnMatchingPlugins(
    const llvm::StringRef pattern,
    std::function<void(const PluginNamespace &plugin_namespace,
                       const std::vector<RegisteredPluginInfo> &plugin_info)>
        action) {
  int num_matching = 0;

  for (const PluginNamespace &plugin_namespace :
       PluginManager::GetPluginNamespaces()) {

    std::vector<RegisteredPluginInfo> matching_plugins;
    for (const RegisteredPluginInfo &plugin_info :
         plugin_namespace.get_info()) {
      if (PluginManager::MatchPluginName(pattern, plugin_namespace,
                                         plugin_info))
        matching_plugins.push_back(plugin_info);
    }

    if (!matching_plugins.empty()) {
      num_matching += matching_plugins.size();
      action(plugin_namespace, matching_plugins);
    }
  }

  return num_matching;
}

// Call the "SetEnable" function for each matching plugins.
// Used to share the majority of the code between the enable
// and disable commands.
int SetEnableOnMatchingPlugins(const llvm::StringRef &pattern,
                               CommandReturnObject &result, bool enabled) {
  return ActOnMatchingPlugins(
      pattern, [&](const PluginNamespace &plugin_namespace,
                   const std::vector<RegisteredPluginInfo> &plugins) {
        result.AppendMessage(plugin_namespace.name);
        for (const auto &plugin : plugins) {
          if (!plugin_namespace.set_enabled(plugin.name, enabled)) {
            result.AppendErrorWithFormat("failed to enable plugin %s.%s",
                                         plugin_namespace.name.data(),
                                         plugin.name.data());
            continue;
          }

          result.AppendMessageWithFormat(
              "  %s %-30s %s\n", enabled ? "[+]" : "[-]", plugin.name.data(),
              plugin.description.data());
        }
      });
}

static std::string ConvertJSONToPrettyString(const llvm::json::Value &json) {
  std::string str;
  llvm::raw_string_ostream os(str);
  os << llvm::formatv("{0:2}", json).str();
  os.flush();
  return str;
}

#define LLDB_OPTIONS_plugin_list
#include "CommandOptions.inc"

// These option definitions are used by the plugin list command.
class PluginListCommandOptions : public Options {
public:
  PluginListCommandOptions() = default;

  ~PluginListCommandOptions() override = default;

  Status SetOptionValue(uint32_t option_idx, llvm::StringRef option_arg,
                        ExecutionContext *execution_context) override {
    Status error;
    const int short_option = m_getopt_table[option_idx].val;

    switch (short_option) {
    case 'j':
      m_json_format = true;
      break;
    default:
      llvm_unreachable("Unimplemented option");
    }

    return error;
  }

  void OptionParsingStarting(ExecutionContext *execution_context) override {
    m_json_format = false;
  }

  llvm::ArrayRef<OptionDefinition> GetDefinitions() override {
    return llvm::ArrayRef(g_plugin_list_options);
  }

  // Instance variables to hold the values for command options.
  bool m_json_format = false;
};
} // namespace

class CommandObjectPluginList : public CommandObjectParsed {
public:
  CommandObjectPluginList(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "plugin list",
                            "Report info about registered LLDB plugins.",
                            nullptr) {
    AddSimpleArgumentList(eArgTypeManagedPlugin);
    SetHelpLong(R"(
Display information about registered plugins.
The plugin information is formatted as shown below:

    <plugin-namespace>
      [+] <plugin-name>                  Plugin #1 description
      [-] <plugin-name>                  Plugin #2 description

An enabled plugin is marked with [+] and a disabled plugin is marked with [-].

Plugins can be listed by namespace and name with:

  plugin list <plugin-namespace>[.<plugin-name>]

Plugins can be listed by namespace alone or with a fully qualified name. When listed
with just a namespace all plugins in that namespace are listed.  When no arguments
are given all plugins are listed.

Examples:
List all plugins

  (lldb) plugin list

List all plugins in the system-runtime namespace

  (lldb) plugin list system-runtime

List only the plugin 'foo' matching a fully qualified name exactly

  (lldb) plugin list system-runtime.foo
)");
  }

  ~CommandObjectPluginList() override = default;

  Options *GetOptions() override { return &m_options; }

  void
  HandleArgumentCompletion(CompletionRequest &request,
                           OptionElementVector &opt_element_vector) override {
    lldb_private::CommandCompletions::InvokeCommonCompletionCallbacks(
        GetCommandInterpreter(), lldb::eManagedPluginCompletion, request,
        nullptr);
  }

protected:
  void DoExecute(Args &command, CommandReturnObject &result) override {
    size_t argc = command.GetArgumentCount();
    result.SetStatus(eReturnStatusSuccessFinishResult);

    // Create a temporary vector to hold the patterns to simplify the logic
    // for the case when the user passes no patterns
    std::vector<llvm::StringRef> patterns;
    patterns.reserve(argc == 0 ? 1 : argc);
    if (argc == 0)
      patterns.push_back("");
    else
      for (size_t i = 0; i < argc; ++i)
        patterns.push_back(command[i].ref());

    if (m_options.m_json_format)
      OutputJsonFormat(patterns, result);
    else
      OutputTextFormat(patterns, result);
  }

private:
  void OutputJsonFormat(const std::vector<llvm::StringRef> &patterns,
                        CommandReturnObject &result) {
    llvm::json::Object obj;
    bool found_empty = false;
    for (const llvm::StringRef pattern : patterns) {
      llvm::json::Object pat_obj = PluginManager::GetJSON(pattern);
      if (pat_obj.empty()) {
        found_empty = true;
        result.AppendErrorWithFormat(
            "Found no matching plugins for pattern '%s'", pattern.data());
        break;
      }
      for (auto &entry : pat_obj) {
        obj[entry.first] = std::move(entry.second);
      }
    }
    if (!found_empty) {
      result.AppendMessage(ConvertJSONToPrettyString(std::move(obj)));
    }
  }

  void OutputTextFormat(const std::vector<llvm::StringRef> &patterns,
                        CommandReturnObject &result) {
    for (const llvm::StringRef pattern : patterns) {
      int num_matching = ActOnMatchingPlugins(
          pattern, [&](const PluginNamespace &plugin_namespace,
                       const std::vector<RegisteredPluginInfo> &plugins) {
            result.AppendMessage(plugin_namespace.name);
            for (auto &plugin : plugins) {
              result.AppendMessageWithFormat(
                  "  %s %-30s %s\n", plugin.enabled ? "[+]" : "[-]",
                  plugin.name.data(), plugin.description.data());
            }
          });
      if (num_matching == 0) {
        result.AppendErrorWithFormat(
            "Found no matching plugins for pattern '%s'", pattern.data());
        break;
      }
    }
  }

  PluginListCommandOptions m_options;
};

static void DoPluginEnableDisable(Args &command, CommandReturnObject &result,
                                  bool enable) {
  const char *name = enable ? "enable" : "disable";
  size_t argc = command.GetArgumentCount();
  if (argc == 0) {
    result.AppendErrorWithFormat("'plugin %s' requires one or more arguments",
                                 name);
    return;
  }
  result.SetStatus(eReturnStatusSuccessFinishResult);

  for (size_t i = 0; i < argc; ++i) {
    llvm::StringRef pattern = command[i].ref();
    int num_matching = SetEnableOnMatchingPlugins(pattern, result, enable);

    if (num_matching == 0) {
      result.AppendErrorWithFormat(
          "Found no matching plugins to %s for pattern '%s'", name,
          pattern.data());
      break;
    }
  }
}

class CommandObjectPluginEnable : public CommandObjectParsed {
public:
  CommandObjectPluginEnable(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "plugin enable",
                            "Enable registered LLDB plugins.", nullptr) {
    AddSimpleArgumentList(eArgTypeManagedPlugin);
  }

  void
  HandleArgumentCompletion(CompletionRequest &request,
                           OptionElementVector &opt_element_vector) override {
    lldb_private::CommandCompletions::InvokeCommonCompletionCallbacks(
        GetCommandInterpreter(), lldb::eManagedPluginCompletion, request,
        nullptr);
  }

  ~CommandObjectPluginEnable() override = default;

protected:
  void DoExecute(Args &command, CommandReturnObject &result) override {
    DoPluginEnableDisable(command, result, /*enable=*/true);
  }
};

class CommandObjectPluginDisable : public CommandObjectParsed {
public:
  CommandObjectPluginDisable(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "plugin disable",
                            "Disable registered LLDB plugins.", nullptr) {
    AddSimpleArgumentList(eArgTypeManagedPlugin);
  }

  void
  HandleArgumentCompletion(CompletionRequest &request,
                           OptionElementVector &opt_element_vector) override {
    lldb_private::CommandCompletions::InvokeCommonCompletionCallbacks(
        GetCommandInterpreter(), lldb::eManagedPluginCompletion, request,
        nullptr);
  }

  ~CommandObjectPluginDisable() override = default;

protected:
  void DoExecute(Args &command, CommandReturnObject &result) override {
    DoPluginEnableDisable(command, result, /*enable=*/false);
  }
};

CommandObjectPlugin::CommandObjectPlugin(CommandInterpreter &interpreter)
    : CommandObjectMultiword(interpreter, "plugin",
                             "Commands for managing LLDB plugins.",
                             "plugin <subcommand> [<subcommand-options>]") {
  LoadSubCommand("load",
                 CommandObjectSP(new CommandObjectPluginLoad(interpreter)));
  LoadSubCommand("list",
                 CommandObjectSP(new CommandObjectPluginList(interpreter)));
  LoadSubCommand("enable",
                 CommandObjectSP(new CommandObjectPluginEnable(interpreter)));
  LoadSubCommand("disable",
                 CommandObjectSP(new CommandObjectPluginDisable(interpreter)));
}

CommandObjectPlugin::~CommandObjectPlugin() = default;
