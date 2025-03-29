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
#define LLDB_OPTIONS_plugin_list
#include "CommandOptions.inc"

// These option definitions are shared by the plugin list/enable/disable
// commands.
class PluginListCommandOptions : public Options {
public:
  PluginListCommandOptions() = default;

  ~PluginListCommandOptions() override = default;

  Status SetOptionValue(uint32_t option_idx, llvm::StringRef option_arg,
                        ExecutionContext *execution_context) override {
    Status error;
    const int short_option = m_getopt_table[option_idx].val;

    switch (short_option) {
    case 'x':
      m_exact_name_match = true;
      break;
    default:
      llvm_unreachable("Unimplemented option");
    }

    return error;
  }

  void OptionParsingStarting(ExecutionContext *execution_context) override {
    m_exact_name_match = false;
  }

  llvm::ArrayRef<OptionDefinition> GetDefinitions() override {
    return llvm::ArrayRef(g_plugin_list_options);
  }

  // Instance variables to hold the values for command options.
  bool m_exact_name_match = false;
};

// Define some data structures to describe known plugin "namespaces".
// The PluginManager is organized into a series of static functions
// that operate on different types of plugin. For example SystemRuntime
// and ObjectFile plugins.
//
// The namespace name is used a prefix when matching plugin names. For example,
// if we have an "elf" plugin in the "object-file" namespace then we will
// match a plugin name pattern against the "object-file.elf" name.
//
// The plugin namespace here is used so we can operate on all the plugins
// of a given type so it is easy to enable or disable them as a group.
using GetPluginInfo = std::function<std::vector<RegisteredPluginInfo>()>;
using SetPluginEnabled = std::function<bool(llvm::StringRef, bool)>;
struct PluginNamespace {
  llvm::StringRef name;
  GetPluginInfo get_info;
  SetPluginEnabled set_enabled;
};

// Currently supported set of plugin namespaces. This will be expanded
// over time.
PluginNamespace PluginNamespaces[] = {
    {"system-runtime", PluginManager::GetSystemRuntimePluginInfo,
     PluginManager::SetSystemRuntimePluginEnabled}};

// Helper function to perform an action on each matching plugin.
// The action callback is given the containing namespace along with plugin info
// for each matching plugin.
static int ActOnMatchingPlugins(
    llvm::GlobPattern pattern,
    std::function<void(const PluginNamespace &plugin_namespace,
                       const std::vector<RegisteredPluginInfo> &plugin_info)>
        action) {
  int num_matching = 0;

  for (const PluginNamespace &plugin_namespace : PluginNamespaces) {
    std::vector<RegisteredPluginInfo> all_plugins = plugin_namespace.get_info();
    std::vector<RegisteredPluginInfo> matching_plugins;
    for (const RegisteredPluginInfo &plugin_info : all_plugins) {
      std::string qualified_name =
          (plugin_namespace.name + "." + plugin_info.name).str();
      if (pattern.match(qualified_name)) {
        matching_plugins.push_back(plugin_info);
      }
    }

    if (!matching_plugins.empty()) {
      num_matching += matching_plugins.size();
      action(plugin_namespace, matching_plugins);
    }
  }

  return num_matching;
}

// Return a string in glob syntax for matching plugins.
static std::string GetPluginNamePatternString(llvm::StringRef user_input,
                                              bool add_default_glob) {
  std::string pattern_str;
  if (user_input.empty())
    pattern_str = "*";
  else
    pattern_str = user_input;

  if (add_default_glob && pattern_str != "*") {
    pattern_str = "*" + pattern_str + "*";
  }

  return pattern_str;
}

// Attempts to create a glob pattern for a plugin name based on plugin command
// input. Writes an error message to the `result` object if the glob cannot be
// created successfully.
//
// The `glob_storage` is used to hold the string data for the glob pattern. The
// llvm::GlobPattern only contains pointers into the string data so we need a
// stable location that can outlive the glob pattern itself.
std::optional<llvm::GlobPattern>
TryCreatePluginPattern(const char *plugin_command_name, const Args &command,
                       const PluginListCommandOptions &options,
                       CommandReturnObject &result, std::string &glob_storage) {
  size_t argc = command.GetArgumentCount();
  if (argc > 1) {
    result.AppendErrorWithFormat("'%s' requires one argument",
                                 plugin_command_name);
    return {};
  }

  llvm::StringRef user_pattern;
  if (argc == 1) {
    user_pattern = command[0].ref();
  }

  glob_storage =
      GetPluginNamePatternString(user_pattern, !options.m_exact_name_match);

  auto glob_pattern = llvm::GlobPattern::create(glob_storage);

  if (auto error = glob_pattern.takeError()) {
    std::string error_message =
        (llvm::Twine("Invalid plugin glob pattern: '") + glob_storage +
         "': " + llvm::toString(std::move(error)))
            .str();
    result.AppendError(error_message);
    return {};
  }

  return *glob_pattern;
}

// Call the "SetEnable" function for each matching plugins.
// Used to share the majority of the code between the enable
// and disable commands.
int SetEnableOnMatchingPlugins(const llvm::GlobPattern &pattern,
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
    AddSimpleArgumentList(eArgTypePlugin);
    SetHelpLong(R"(
Display information about registered plugins.
The plugin information is formatted as shown below

    <plugin-namespace>
      [+] <plugin-name>                  Plugin #1 description
      [-] <plugin-name>                  Plugin #2 description

An enabled plugin is marked with [+] and a disabled plugin is marked with [-].

Selecting plugins
------------------
plugin list [<plugin-namespace>.][<plugin-name>]

Plugin names are specified using glob patterns. The pattern will be matched
against the plugins fully qualified name, which is composed of the namespace,
followed by a '.', followed by the plugin name.

When no arguments are given the plugin selection string is the wildcard '*'.
By default wildcards are added around the input to enable searching by
substring. You can prevent these implicit wild cards by using the
-x flag.

Examples
-----------------
List all plugins in the system-runtime namespace

  (lldb) plugin list system-runtime.*

List all plugins containing the string foo

  (lldb) plugin list foo

This is equivalent to

  (lldb) plugin list *foo*

List only a plugin matching a fully qualified name exactly

  (lldb) plugin list -x system-runtime.foo
)");
  }

  ~CommandObjectPluginList() override = default;

  Options *GetOptions() override { return &m_options; }

protected:
  void DoExecute(Args &command, CommandReturnObject &result) override {
    std::string glob_storage;
    std::optional<llvm::GlobPattern> plugin_glob = TryCreatePluginPattern(
        "plugin list", command, m_options, result, glob_storage);

    if (!plugin_glob) {
      assert(!result.Succeeded());
      return;
    }

    int num_matching = ActOnMatchingPlugins(
        *plugin_glob, [&](const PluginNamespace &plugin_namespace,
                          const std::vector<RegisteredPluginInfo> &plugins) {
          result.AppendMessage(plugin_namespace.name);
          for (auto &plugin : plugins) {
            result.AppendMessageWithFormat(
                "  %s %-30s %s\n", plugin.enabled ? "[+]" : "[-]",
                plugin.name.data(), plugin.description.data());
          }
        });

    if (num_matching == 0) {
      result.AppendErrorWithFormat("Found no matching plugins");
    }
  }

  PluginListCommandOptions m_options;
};

class CommandObjectPluginEnable : public CommandObjectParsed {
public:
  CommandObjectPluginEnable(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "plugin enable",
                            "Enable registered LLDB plugins.", nullptr) {
    AddSimpleArgumentList(eArgTypePlugin);
  }

  ~CommandObjectPluginEnable() override = default;

  Options *GetOptions() override { return &m_options; }

protected:
  void DoExecute(Args &command, CommandReturnObject &result) override {
    std::string glob_storage;
    std::optional<llvm::GlobPattern> plugin_glob = TryCreatePluginPattern(
        "plugin enable", command, m_options, result, glob_storage);

    if (!plugin_glob) {
      assert(!result.Succeeded());
      return;
    }

    int num_matching = SetEnableOnMatchingPlugins(*plugin_glob, result, true);

    if (num_matching == 0) {
      result.AppendErrorWithFormat("Found no matching plugins to enable");
    }
  }

  PluginListCommandOptions m_options;
};

class CommandObjectPluginDisable : public CommandObjectParsed {
public:
  CommandObjectPluginDisable(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "plugin disable",
                            "Disable registered LLDB plugins.", nullptr) {
    AddSimpleArgumentList(eArgTypePlugin);
  }

  ~CommandObjectPluginDisable() override = default;

  Options *GetOptions() override { return &m_options; }

protected:
  void DoExecute(Args &command, CommandReturnObject &result) override {
    std::string glob_storage;
    std::optional<llvm::GlobPattern> plugin_glob = TryCreatePluginPattern(
        "plugin disable", command, m_options, result, glob_storage);

    if (!plugin_glob) {
      assert(!result.Succeeded());
      return;
    }

    int num_matching = SetEnableOnMatchingPlugins(*plugin_glob, result, false);

    if (num_matching == 0) {
      result.AppendErrorWithFormat("Found no matching plugins to disable");
    }
  }

  PluginListCommandOptions m_options;
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
