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
#include "llvm/Support/GlobPattern.h"

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
    const bool match_namespace =
        pattern.empty() || pattern == plugin_namespace.name;

    std::vector<RegisteredPluginInfo> matching_plugins;
    for (const RegisteredPluginInfo &plugin_info :
         plugin_namespace.get_info()) {

      // If we match the namespace, we can skip the plugin name check.
      bool match_qualified_name = false;
      if (!match_namespace) {
        std::string qualified_name =
            (plugin_namespace.name + "." + plugin_info.name).str();
        match_qualified_name = pattern == qualified_name;
      }

      if (match_namespace || match_qualified_name)
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

protected:
  void DoExecute(Args &command, CommandReturnObject &result) override {
    size_t argc = command.GetArgumentCount();
    if (argc > 1) {
      result.AppendError("'plugin load' requires one argument");
      return;
    }
    llvm::StringRef pattern = argc ? command[0].ref() : "";
    result.SetStatus(eReturnStatusSuccessFinishResult);

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

    if (num_matching == 0)
      result.AppendErrorWithFormat("Found no matching plugins");
  }
};

class CommandObjectPluginEnable : public CommandObjectParsed {
public:
  CommandObjectPluginEnable(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "plugin enable",
                            "Enable registered LLDB plugins.", nullptr) {
    AddSimpleArgumentList(eArgTypeManagedPlugin);
  }

  ~CommandObjectPluginEnable() override = default;

protected:
  void DoExecute(Args &command, CommandReturnObject &result) override {
    size_t argc = command.GetArgumentCount();
    if (argc > 1) {
      result.AppendError("'plugin enable' requires one argument");
      return;
    }
    llvm::StringRef pattern = argc ? command[0].ref() : "";
    result.SetStatus(eReturnStatusSuccessFinishResult);

    int num_matching = SetEnableOnMatchingPlugins(pattern, result, true);

    if (num_matching == 0)
      result.AppendErrorWithFormat("Found no matching plugins to enable");
  }
};

class CommandObjectPluginDisable : public CommandObjectParsed {
public:
  CommandObjectPluginDisable(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "plugin disable",
                            "Disable registered LLDB plugins.", nullptr) {
    AddSimpleArgumentList(eArgTypeManagedPlugin);
  }

  ~CommandObjectPluginDisable() override = default;

protected:
  void DoExecute(Args &command, CommandReturnObject &result) override {
    size_t argc = command.GetArgumentCount();
    if (argc > 1) {
      result.AppendError("'plugin disable' requires one argument");
      return;
    }
    llvm::StringRef pattern = argc ? command[0].ref() : "";
    result.SetStatus(eReturnStatusSuccessFinishResult);

    int num_matching = SetEnableOnMatchingPlugins(pattern, result, false);

    if (num_matching == 0)
      result.AppendErrorWithFormat("Found no matching plugins to disable");
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
