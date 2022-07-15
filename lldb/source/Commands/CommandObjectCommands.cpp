//===-- CommandObjectCommands.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CommandObjectCommands.h"
#include "CommandObjectHelp.h"
#include "CommandObjectRegexCommand.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/IOHandler.h"
#include "lldb/Interpreter/CommandHistory.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandOptionArgumentTable.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/OptionArgParser.h"
#include "lldb/Interpreter/OptionValueBoolean.h"
#include "lldb/Interpreter/OptionValueString.h"
#include "lldb/Interpreter/OptionValueUInt64.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/Utility/Args.h"
#include "lldb/Utility/StringList.h"
#include "llvm/ADT/StringRef.h"

using namespace lldb;
using namespace lldb_private;

// CommandObjectCommandsSource

#define LLDB_OPTIONS_source
#include "CommandOptions.inc"

class CommandObjectCommandsSource : public CommandObjectParsed {
public:
  CommandObjectCommandsSource(CommandInterpreter &interpreter)
      : CommandObjectParsed(
            interpreter, "command source",
            "Read and execute LLDB commands from the file <filename>.",
            nullptr) {
    CommandArgumentEntry arg;
    CommandArgumentData file_arg;

    // Define the first (and only) variant of this arg.
    file_arg.arg_type = eArgTypeFilename;
    file_arg.arg_repetition = eArgRepeatPlain;

    // There is only one variant this argument could be; put it into the
    // argument entry.
    arg.push_back(file_arg);

    // Push the data for the first argument into the m_arguments vector.
    m_arguments.push_back(arg);
  }

  ~CommandObjectCommandsSource() override = default;

  llvm::Optional<std::string> GetRepeatCommand(Args &current_command_args,
                                               uint32_t index) override {
    return std::string("");
  }

  void
  HandleArgumentCompletion(CompletionRequest &request,
                           OptionElementVector &opt_element_vector) override {
    CommandCompletions::InvokeCommonCompletionCallbacks(
        GetCommandInterpreter(), CommandCompletions::eDiskFileCompletion,
        request, nullptr);
  }

  Options *GetOptions() override { return &m_options; }

protected:
  class CommandOptions : public Options {
  public:
    CommandOptions()
        : m_stop_on_error(true), m_silent_run(false), m_stop_on_continue(true),
          m_cmd_relative_to_command_file(false) {}

    ~CommandOptions() override = default;

    Status SetOptionValue(uint32_t option_idx, llvm::StringRef option_arg,
                          ExecutionContext *execution_context) override {
      Status error;
      const int short_option = m_getopt_table[option_idx].val;

      switch (short_option) {
      case 'e':
        error = m_stop_on_error.SetValueFromString(option_arg);
        break;

      case 'c':
        error = m_stop_on_continue.SetValueFromString(option_arg);
        break;

      case 'C':
        m_cmd_relative_to_command_file = true;
        break;

      case 's':
        error = m_silent_run.SetValueFromString(option_arg);
        break;

      default:
        llvm_unreachable("Unimplemented option");
      }

      return error;
    }

    void OptionParsingStarting(ExecutionContext *execution_context) override {
      m_stop_on_error.Clear();
      m_silent_run.Clear();
      m_stop_on_continue.Clear();
      m_cmd_relative_to_command_file.Clear();
    }

    llvm::ArrayRef<OptionDefinition> GetDefinitions() override {
      return llvm::makeArrayRef(g_source_options);
    }

    // Instance variables to hold the values for command options.

    OptionValueBoolean m_stop_on_error;
    OptionValueBoolean m_silent_run;
    OptionValueBoolean m_stop_on_continue;
    OptionValueBoolean m_cmd_relative_to_command_file;
  };

  bool DoExecute(Args &command, CommandReturnObject &result) override {
    if (command.GetArgumentCount() != 1) {
      result.AppendErrorWithFormat(
          "'%s' takes exactly one executable filename argument.\n",
          GetCommandName().str().c_str());
      return false;
    }

    FileSpec source_dir = {};
    if (m_options.m_cmd_relative_to_command_file) {
      source_dir = GetDebugger().GetCommandInterpreter().GetCurrentSourceDir();
      if (!source_dir) {
        result.AppendError("command source -C can only be specified "
                           "from a command file");
        result.SetStatus(eReturnStatusFailed);
        return false;
      }
    }

    FileSpec cmd_file(command[0].ref());
    if (source_dir) {
      // Prepend the source_dir to the cmd_file path:
      if (!cmd_file.IsRelative()) {
        result.AppendError("command source -C can only be used "
                           "with a relative path.");
        result.SetStatus(eReturnStatusFailed);
        return false;
      }
      cmd_file.MakeAbsolute(source_dir);
    }

    FileSystem::Instance().Resolve(cmd_file);

    CommandInterpreterRunOptions options;
    // If any options were set, then use them
    if (m_options.m_stop_on_error.OptionWasSet() ||
        m_options.m_silent_run.OptionWasSet() ||
        m_options.m_stop_on_continue.OptionWasSet()) {
      if (m_options.m_stop_on_continue.OptionWasSet())
        options.SetStopOnContinue(
            m_options.m_stop_on_continue.GetCurrentValue());

      if (m_options.m_stop_on_error.OptionWasSet())
        options.SetStopOnError(m_options.m_stop_on_error.GetCurrentValue());

      // Individual silent setting is override for global command echo settings.
      if (m_options.m_silent_run.GetCurrentValue()) {
        options.SetSilent(true);
      } else {
        options.SetPrintResults(true);
        options.SetPrintErrors(true);
        options.SetEchoCommands(m_interpreter.GetEchoCommands());
        options.SetEchoCommentCommands(m_interpreter.GetEchoCommentCommands());
      }
    }

    m_interpreter.HandleCommandsFromFile(cmd_file, options, result);
    return result.Succeeded();
  }

  CommandOptions m_options;
};

#pragma mark CommandObjectCommandsAlias
// CommandObjectCommandsAlias

#define LLDB_OPTIONS_alias
#include "CommandOptions.inc"

static const char *g_python_command_instructions =
    "Enter your Python command(s). Type 'DONE' to end.\n"
    "You must define a Python function with this signature:\n"
    "def my_command_impl(debugger, args, result, internal_dict):\n";

class CommandObjectCommandsAlias : public CommandObjectRaw {
protected:
  class CommandOptions : public OptionGroup {
  public:
    CommandOptions() = default;

    ~CommandOptions() override = default;

    llvm::ArrayRef<OptionDefinition> GetDefinitions() override {
      return llvm::makeArrayRef(g_alias_options);
    }

    Status SetOptionValue(uint32_t option_idx, llvm::StringRef option_value,
                          ExecutionContext *execution_context) override {
      Status error;

      const int short_option = GetDefinitions()[option_idx].short_option;
      std::string option_str(option_value);

      switch (short_option) {
      case 'h':
        m_help.SetCurrentValue(option_str);
        m_help.SetOptionWasSet();
        break;

      case 'H':
        m_long_help.SetCurrentValue(option_str);
        m_long_help.SetOptionWasSet();
        break;

      default:
        llvm_unreachable("Unimplemented option");
      }

      return error;
    }

    void OptionParsingStarting(ExecutionContext *execution_context) override {
      m_help.Clear();
      m_long_help.Clear();
    }

    OptionValueString m_help;
    OptionValueString m_long_help;
  };

  OptionGroupOptions m_option_group;
  CommandOptions m_command_options;

public:
  Options *GetOptions() override { return &m_option_group; }

  CommandObjectCommandsAlias(CommandInterpreter &interpreter)
      : CommandObjectRaw(
            interpreter, "command alias",
            "Define a custom command in terms of an existing command.") {
    m_option_group.Append(&m_command_options);
    m_option_group.Finalize();

    SetHelpLong(
        "'alias' allows the user to create a short-cut or abbreviation for long \
commands, multi-word commands, and commands that take particular options.  \
Below are some simple examples of how one might use the 'alias' command:"
        R"(

(lldb) command alias sc script

    Creates the abbreviation 'sc' for the 'script' command.

(lldb) command alias bp breakpoint

)"
        "    Creates the abbreviation 'bp' for the 'breakpoint' command.  Since \
breakpoint commands are two-word commands, the user would still need to \
enter the second word after 'bp', e.g. 'bp enable' or 'bp delete'."
        R"(

(lldb) command alias bpl breakpoint list

    Creates the abbreviation 'bpl' for the two-word command 'breakpoint list'.

)"
        "An alias can include some options for the command, with the values either \
filled in at the time the alias is created, or specified as positional \
arguments, to be filled in when the alias is invoked.  The following example \
shows how to create aliases with options:"
        R"(

(lldb) command alias bfl breakpoint set -f %1 -l %2

)"
        "    Creates the abbreviation 'bfl' (for break-file-line), with the -f and -l \
options already part of the alias.  So if the user wants to set a breakpoint \
by file and line without explicitly having to use the -f and -l options, the \
user can now use 'bfl' instead.  The '%1' and '%2' are positional placeholders \
for the actual arguments that will be passed when the alias command is used.  \
The number in the placeholder refers to the position/order the actual value \
occupies when the alias is used.  All the occurrences of '%1' in the alias \
will be replaced with the first argument, all the occurrences of '%2' in the \
alias will be replaced with the second argument, and so on.  This also allows \
actual arguments to be used multiple times within an alias (see 'process \
launch' example below)."
        R"(

)"
        "Note: the positional arguments must substitute as whole words in the resultant \
command, so you can't at present do something like this to append the file extension \
\".cpp\":"
        R"(

(lldb) command alias bcppfl breakpoint set -f %1.cpp -l %2

)"
        "For more complex aliasing, use the \"command regex\" command instead.  In the \
'bfl' case above, the actual file value will be filled in with the first argument \
following 'bfl' and the actual line number value will be filled in with the second \
argument.  The user would use this alias as follows:"
        R"(

(lldb) command alias bfl breakpoint set -f %1 -l %2
(lldb) bfl my-file.c 137

This would be the same as if the user had entered 'breakpoint set -f my-file.c -l 137'.

Another example:

(lldb) command alias pltty process launch -s -o %1 -e %1
(lldb) pltty /dev/tty0

    Interpreted as 'process launch -s -o /dev/tty0 -e /dev/tty0'

)"
        "If the user always wanted to pass the same value to a particular option, the \
alias could be defined with that value directly in the alias as a constant, \
rather than using a positional placeholder:"
        R"(

(lldb) command alias bl3 breakpoint set -f %1 -l 3

    Always sets a breakpoint on line 3 of whatever file is indicated.)");

    CommandArgumentEntry arg1;
    CommandArgumentEntry arg2;
    CommandArgumentEntry arg3;
    CommandArgumentData alias_arg;
    CommandArgumentData cmd_arg;
    CommandArgumentData options_arg;

    // Define the first (and only) variant of this arg.
    alias_arg.arg_type = eArgTypeAliasName;
    alias_arg.arg_repetition = eArgRepeatPlain;

    // There is only one variant this argument could be; put it into the
    // argument entry.
    arg1.push_back(alias_arg);

    // Define the first (and only) variant of this arg.
    cmd_arg.arg_type = eArgTypeCommandName;
    cmd_arg.arg_repetition = eArgRepeatPlain;

    // There is only one variant this argument could be; put it into the
    // argument entry.
    arg2.push_back(cmd_arg);

    // Define the first (and only) variant of this arg.
    options_arg.arg_type = eArgTypeAliasOptions;
    options_arg.arg_repetition = eArgRepeatOptional;

    // There is only one variant this argument could be; put it into the
    // argument entry.
    arg3.push_back(options_arg);

    // Push the data for the first argument into the m_arguments vector.
    m_arguments.push_back(arg1);
    m_arguments.push_back(arg2);
    m_arguments.push_back(arg3);
  }

  ~CommandObjectCommandsAlias() override = default;

protected:
  bool DoExecute(llvm::StringRef raw_command_line,
                 CommandReturnObject &result) override {
    if (raw_command_line.empty()) {
      result.AppendError("'command alias' requires at least two arguments");
      return false;
    }

    ExecutionContext exe_ctx = GetCommandInterpreter().GetExecutionContext();
    m_option_group.NotifyOptionParsingStarting(&exe_ctx);

    OptionsWithRaw args_with_suffix(raw_command_line);

    if (args_with_suffix.HasArgs())
      if (!ParseOptionsAndNotify(args_with_suffix.GetArgs(), result,
                                 m_option_group, exe_ctx))
        return false;

    llvm::StringRef raw_command_string = args_with_suffix.GetRawPart();
    Args args(raw_command_string);

    if (args.GetArgumentCount() < 2) {
      result.AppendError("'command alias' requires at least two arguments");
      return false;
    }

    // Get the alias command.

    auto alias_command = args[0].ref();
    if (alias_command.startswith("-")) {
      result.AppendError("aliases starting with a dash are not supported");
      if (alias_command == "--help" || alias_command == "--long-help") {
        result.AppendWarning("if trying to pass options to 'command alias' add "
                             "a -- at the end of the options");
      }
      return false;
    }

    // Strip the new alias name off 'raw_command_string'  (leave it on args,
    // which gets passed to 'Execute', which does the stripping itself.
    size_t pos = raw_command_string.find(alias_command);
    if (pos == 0) {
      raw_command_string = raw_command_string.substr(alias_command.size());
      pos = raw_command_string.find_first_not_of(' ');
      if ((pos != std::string::npos) && (pos > 0))
        raw_command_string = raw_command_string.substr(pos);
    } else {
      result.AppendError("Error parsing command string.  No alias created.");
      return false;
    }

    // Verify that the command is alias-able.
    if (m_interpreter.CommandExists(alias_command)) {
      result.AppendErrorWithFormat(
          "'%s' is a permanent debugger command and cannot be redefined.\n",
          args[0].c_str());
      return false;
    }

    if (m_interpreter.UserMultiwordCommandExists(alias_command)) {
      result.AppendErrorWithFormat(
          "'%s' is a user container command and cannot be overwritten.\n"
          "Delete it first with 'command container delete'\n",
          args[0].c_str());
      return false;
    }

    // Get CommandObject that is being aliased. The command name is read from
    // the front of raw_command_string. raw_command_string is returned with the
    // name of the command object stripped off the front.
    llvm::StringRef original_raw_command_string = raw_command_string;
    CommandObject *cmd_obj =
        m_interpreter.GetCommandObjectForCommand(raw_command_string);

    if (!cmd_obj) {
      result.AppendErrorWithFormat("invalid command given to 'command alias'. "
                                   "'%s' does not begin with a valid command."
                                   "  No alias created.",
                                   original_raw_command_string.str().c_str());
      return false;
    } else if (!cmd_obj->WantsRawCommandString()) {
      // Note that args was initialized with the original command, and has not
      // been updated to this point. Therefore can we pass it to the version of
      // Execute that does not need/expect raw input in the alias.
      return HandleAliasingNormalCommand(args, result);
    } else {
      return HandleAliasingRawCommand(alias_command, raw_command_string,
                                      *cmd_obj, result);
    }
    return result.Succeeded();
  }

  bool HandleAliasingRawCommand(llvm::StringRef alias_command,
                                llvm::StringRef raw_command_string,
                                CommandObject &cmd_obj,
                                CommandReturnObject &result) {
    // Verify & handle any options/arguments passed to the alias command

    OptionArgVectorSP option_arg_vector_sp =
        OptionArgVectorSP(new OptionArgVector);

    const bool include_aliases = true;
    if (CommandObjectSP cmd_obj_sp = m_interpreter.GetCommandSPExact(
            cmd_obj.GetCommandName(), include_aliases)) {
      if (m_interpreter.AliasExists(alias_command) ||
          m_interpreter.UserCommandExists(alias_command)) {
        result.AppendWarningWithFormat(
            "Overwriting existing definition for '%s'.\n",
            alias_command.str().c_str());
      }
      if (CommandAlias *alias = m_interpreter.AddAlias(
              alias_command, cmd_obj_sp, raw_command_string)) {
        if (m_command_options.m_help.OptionWasSet())
          alias->SetHelp(m_command_options.m_help.GetCurrentValue());
        if (m_command_options.m_long_help.OptionWasSet())
          alias->SetHelpLong(m_command_options.m_long_help.GetCurrentValue());
        result.SetStatus(eReturnStatusSuccessFinishNoResult);
      } else {
        result.AppendError("Unable to create requested alias.\n");
      }

    } else {
      result.AppendError("Unable to create requested alias.\n");
    }

    return result.Succeeded();
  }

  bool HandleAliasingNormalCommand(Args &args, CommandReturnObject &result) {
    size_t argc = args.GetArgumentCount();

    if (argc < 2) {
      result.AppendError("'command alias' requires at least two arguments");
      return false;
    }

    // Save these in std::strings since we're going to shift them off.
    const std::string alias_command(std::string(args[0].ref()));
    const std::string actual_command(std::string(args[1].ref()));

    args.Shift(); // Shift the alias command word off the argument vector.
    args.Shift(); // Shift the old command word off the argument vector.

    // Verify that the command is alias'able, and get the appropriate command
    // object.

    if (m_interpreter.CommandExists(alias_command)) {
      result.AppendErrorWithFormat(
          "'%s' is a permanent debugger command and cannot be redefined.\n",
          alias_command.c_str());
      return false;
    }

    if (m_interpreter.UserMultiwordCommandExists(alias_command)) {
      result.AppendErrorWithFormat(
          "'%s' is user container command and cannot be overwritten.\n"
          "Delete it first with 'command container delete'",
          alias_command.c_str());
      return false;
    }

    CommandObjectSP command_obj_sp(
        m_interpreter.GetCommandSPExact(actual_command, true));
    CommandObjectSP subcommand_obj_sp;
    bool use_subcommand = false;
    if (!command_obj_sp) {
      result.AppendErrorWithFormat("'%s' is not an existing command.\n",
                                   actual_command.c_str());
      return false;
    }
    CommandObject *cmd_obj = command_obj_sp.get();
    CommandObject *sub_cmd_obj = nullptr;
    OptionArgVectorSP option_arg_vector_sp =
        OptionArgVectorSP(new OptionArgVector);

    while (cmd_obj->IsMultiwordObject() && !args.empty()) {
      auto sub_command = args[0].ref();
      assert(!sub_command.empty());
      subcommand_obj_sp = cmd_obj->GetSubcommandSP(sub_command);
      if (!subcommand_obj_sp) {
        result.AppendErrorWithFormat(
            "'%s' is not a valid sub-command of '%s'.  "
            "Unable to create alias.\n",
            args[0].c_str(), actual_command.c_str());
        return false;
      }

      sub_cmd_obj = subcommand_obj_sp.get();
      use_subcommand = true;
      args.Shift(); // Shift the sub_command word off the argument vector.
      cmd_obj = sub_cmd_obj;
    }

    // Verify & handle any options/arguments passed to the alias command

    std::string args_string;

    if (!args.empty()) {
      CommandObjectSP tmp_sp =
          m_interpreter.GetCommandSPExact(cmd_obj->GetCommandName());
      if (use_subcommand)
        tmp_sp = m_interpreter.GetCommandSPExact(sub_cmd_obj->GetCommandName());

      args.GetCommandString(args_string);
    }

    if (m_interpreter.AliasExists(alias_command) ||
        m_interpreter.UserCommandExists(alias_command)) {
      result.AppendWarningWithFormat(
          "Overwriting existing definition for '%s'.\n", alias_command.c_str());
    }

    if (CommandAlias *alias = m_interpreter.AddAlias(
            alias_command, use_subcommand ? subcommand_obj_sp : command_obj_sp,
            args_string)) {
      if (m_command_options.m_help.OptionWasSet())
        alias->SetHelp(m_command_options.m_help.GetCurrentValue());
      if (m_command_options.m_long_help.OptionWasSet())
        alias->SetHelpLong(m_command_options.m_long_help.GetCurrentValue());
      result.SetStatus(eReturnStatusSuccessFinishNoResult);
    } else {
      result.AppendError("Unable to create requested alias.\n");
      return false;
    }

    return result.Succeeded();
  }
};

#pragma mark CommandObjectCommandsUnalias
// CommandObjectCommandsUnalias

class CommandObjectCommandsUnalias : public CommandObjectParsed {
public:
  CommandObjectCommandsUnalias(CommandInterpreter &interpreter)
      : CommandObjectParsed(
            interpreter, "command unalias",
            "Delete one or more custom commands defined by 'command alias'.",
            nullptr) {
    CommandArgumentEntry arg;
    CommandArgumentData alias_arg;

    // Define the first (and only) variant of this arg.
    alias_arg.arg_type = eArgTypeAliasName;
    alias_arg.arg_repetition = eArgRepeatPlain;

    // There is only one variant this argument could be; put it into the
    // argument entry.
    arg.push_back(alias_arg);

    // Push the data for the first argument into the m_arguments vector.
    m_arguments.push_back(arg);
  }

  ~CommandObjectCommandsUnalias() override = default;

  void
  HandleArgumentCompletion(CompletionRequest &request,
                           OptionElementVector &opt_element_vector) override {
    if (!m_interpreter.HasCommands() || request.GetCursorIndex() != 0)
      return;

    for (const auto &ent : m_interpreter.GetAliases()) {
      request.TryCompleteCurrentArg(ent.first, ent.second->GetHelp());
    }
  }

protected:
  bool DoExecute(Args &args, CommandReturnObject &result) override {
    CommandObject::CommandMap::iterator pos;
    CommandObject *cmd_obj;

    if (args.empty()) {
      result.AppendError("must call 'unalias' with a valid alias");
      return false;
    }

    auto command_name = args[0].ref();
    cmd_obj = m_interpreter.GetCommandObject(command_name);
    if (!cmd_obj) {
      result.AppendErrorWithFormat(
          "'%s' is not a known command.\nTry 'help' to see a "
          "current list of commands.\n",
          args[0].c_str());
      return false;
    }

    if (m_interpreter.CommandExists(command_name)) {
      if (cmd_obj->IsRemovable()) {
        result.AppendErrorWithFormat(
            "'%s' is not an alias, it is a debugger command which can be "
            "removed using the 'command delete' command.\n",
            args[0].c_str());
      } else {
        result.AppendErrorWithFormat(
            "'%s' is a permanent debugger command and cannot be removed.\n",
            args[0].c_str());
      }
      return false;
    }

    if (!m_interpreter.RemoveAlias(command_name)) {
      if (m_interpreter.AliasExists(command_name))
        result.AppendErrorWithFormat(
            "Error occurred while attempting to unalias '%s'.\n",
            args[0].c_str());
      else
        result.AppendErrorWithFormat("'%s' is not an existing alias.\n",
                                     args[0].c_str());
      return false;
    }

    result.SetStatus(eReturnStatusSuccessFinishNoResult);
    return result.Succeeded();
  }
};

#pragma mark CommandObjectCommandsDelete
// CommandObjectCommandsDelete

class CommandObjectCommandsDelete : public CommandObjectParsed {
public:
  CommandObjectCommandsDelete(CommandInterpreter &interpreter)
      : CommandObjectParsed(
            interpreter, "command delete",
            "Delete one or more custom commands defined by 'command regex'.",
            nullptr) {
    CommandArgumentEntry arg;
    CommandArgumentData alias_arg;

    // Define the first (and only) variant of this arg.
    alias_arg.arg_type = eArgTypeCommandName;
    alias_arg.arg_repetition = eArgRepeatPlain;

    // There is only one variant this argument could be; put it into the
    // argument entry.
    arg.push_back(alias_arg);

    // Push the data for the first argument into the m_arguments vector.
    m_arguments.push_back(arg);
  }

  ~CommandObjectCommandsDelete() override = default;

  void
  HandleArgumentCompletion(CompletionRequest &request,
                           OptionElementVector &opt_element_vector) override {
    if (!m_interpreter.HasCommands() || request.GetCursorIndex() != 0)
      return;

    for (const auto &ent : m_interpreter.GetCommands()) {
      if (ent.second->IsRemovable())
        request.TryCompleteCurrentArg(ent.first, ent.second->GetHelp());
    }
  }

protected:
  bool DoExecute(Args &args, CommandReturnObject &result) override {
    CommandObject::CommandMap::iterator pos;

    if (args.empty()) {
      result.AppendErrorWithFormat("must call '%s' with one or more valid user "
                                   "defined regular expression command names",
                                   GetCommandName().str().c_str());
      return false;
    }

    auto command_name = args[0].ref();
    if (!m_interpreter.CommandExists(command_name)) {
      StreamString error_msg_stream;
      const bool generate_upropos = true;
      const bool generate_type_lookup = false;
      CommandObjectHelp::GenerateAdditionalHelpAvenuesMessage(
          &error_msg_stream, command_name, llvm::StringRef(), llvm::StringRef(),
          generate_upropos, generate_type_lookup);
      result.AppendError(error_msg_stream.GetString());
      return false;
    }

    if (!m_interpreter.RemoveCommand(command_name)) {
      result.AppendErrorWithFormat(
          "'%s' is a permanent debugger command and cannot be removed.\n",
          args[0].c_str());
      return false;
    }

    result.SetStatus(eReturnStatusSuccessFinishNoResult);
    return true;
  }
};

// CommandObjectCommandsAddRegex

#define LLDB_OPTIONS_regex
#include "CommandOptions.inc"

#pragma mark CommandObjectCommandsAddRegex

class CommandObjectCommandsAddRegex : public CommandObjectParsed,
                                      public IOHandlerDelegateMultiline {
public:
  CommandObjectCommandsAddRegex(CommandInterpreter &interpreter)
      : CommandObjectParsed(
            interpreter, "command regex",
            "Define a custom command in terms of "
            "existing commands by matching "
            "regular expressions.",
            "command regex <cmd-name> [s/<regex>/<subst>/ ...]"),
        IOHandlerDelegateMultiline("",
                                   IOHandlerDelegate::Completion::LLDBCommand) {
    SetHelpLong(
        R"(
)"
        "This command allows the user to create powerful regular expression commands \
with substitutions. The regular expressions and substitutions are specified \
using the regular expression substitution format of:"
        R"(

    s/<regex>/<subst>/

)"
        "<regex> is a regular expression that can use parenthesis to capture regular \
expression input and substitute the captured matches in the output using %1 \
for the first match, %2 for the second, and so on."
        R"(

)"
        "The regular expressions can all be specified on the command line if more than \
one argument is provided. If just the command name is provided on the command \
line, then the regular expressions and substitutions can be entered on separate \
lines, followed by an empty line to terminate the command definition."
        R"(

EXAMPLES

)"
        "The following example will define a regular expression command named 'f' that \
will call 'finish' if there are no arguments, or 'frame select <frame-idx>' if \
a number follows 'f':"
        R"(

    (lldb) command regex f s/^$/finish/ 's/([0-9]+)/frame select %1/')");
    CommandArgumentData thread_arg{eArgTypeSEDStylePair, eArgRepeatOptional};
    m_arguments.push_back({thread_arg});
  }

  ~CommandObjectCommandsAddRegex() override = default;

protected:
  void IOHandlerActivated(IOHandler &io_handler, bool interactive) override {
    StreamFileSP output_sp(io_handler.GetOutputStreamFileSP());
    if (output_sp && interactive) {
      output_sp->PutCString("Enter one or more sed substitution commands in "
                            "the form: 's/<regex>/<subst>/'.\nTerminate the "
                            "substitution list with an empty line.\n");
      output_sp->Flush();
    }
  }

  void IOHandlerInputComplete(IOHandler &io_handler,
                              std::string &data) override {
    io_handler.SetIsDone(true);
    if (m_regex_cmd_up) {
      StringList lines;
      if (lines.SplitIntoLines(data)) {
        bool check_only = false;
        for (const std::string &line : lines) {
          Status error = AppendRegexSubstitution(line, check_only);
          if (error.Fail()) {
            if (!GetDebugger().GetCommandInterpreter().GetBatchCommandMode()) {
              StreamSP out_stream = GetDebugger().GetAsyncOutputStream();
              out_stream->Printf("error: %s\n", error.AsCString());
            }
          }
        }
      }
      if (m_regex_cmd_up->HasRegexEntries()) {
        CommandObjectSP cmd_sp(m_regex_cmd_up.release());
        m_interpreter.AddCommand(cmd_sp->GetCommandName(), cmd_sp, true);
      }
    }
  }

  bool DoExecute(Args &command, CommandReturnObject &result) override {
    const size_t argc = command.GetArgumentCount();
    if (argc == 0) {
      result.AppendError("usage: 'command regex <command-name> "
                         "[s/<regex1>/<subst1>/ s/<regex2>/<subst2>/ ...]'\n");
      return false;
    }

    Status error;
    auto name = command[0].ref();
    m_regex_cmd_up = std::make_unique<CommandObjectRegexCommand>(
        m_interpreter, name, m_options.GetHelp(), m_options.GetSyntax(), 10, 0,
        true);

    if (argc == 1) {
      Debugger &debugger = GetDebugger();
      bool color_prompt = debugger.GetUseColor();
      const bool multiple_lines = true; // Get multiple lines
      IOHandlerSP io_handler_sp(new IOHandlerEditline(
          debugger, IOHandler::Type::Other,
          "lldb-regex",          // Name of input reader for history
          llvm::StringRef("> "), // Prompt
          llvm::StringRef(),     // Continuation prompt
          multiple_lines, color_prompt,
          0, // Don't show line numbers
          *this, nullptr));

      if (io_handler_sp) {
        debugger.RunIOHandlerAsync(io_handler_sp);
        result.SetStatus(eReturnStatusSuccessFinishNoResult);
      }
    } else {
      for (auto &entry : command.entries().drop_front()) {
        bool check_only = false;
        error = AppendRegexSubstitution(entry.ref(), check_only);
        if (error.Fail())
          break;
      }

      if (error.Success()) {
        AddRegexCommandToInterpreter();
      }
    }
    if (error.Fail()) {
      result.AppendError(error.AsCString());
    }

    return result.Succeeded();
  }

  Status AppendRegexSubstitution(const llvm::StringRef &regex_sed,
                                 bool check_only) {
    Status error;

    if (!m_regex_cmd_up) {
      error.SetErrorStringWithFormat(
          "invalid regular expression command object for: '%.*s'",
          (int)regex_sed.size(), regex_sed.data());
      return error;
    }

    size_t regex_sed_size = regex_sed.size();

    if (regex_sed_size <= 1) {
      error.SetErrorStringWithFormat(
          "regular expression substitution string is too short: '%.*s'",
          (int)regex_sed.size(), regex_sed.data());
      return error;
    }

    if (regex_sed[0] != 's') {
      error.SetErrorStringWithFormat("regular expression substitution string "
                                     "doesn't start with 's': '%.*s'",
                                     (int)regex_sed.size(), regex_sed.data());
      return error;
    }
    const size_t first_separator_char_pos = 1;
    // use the char that follows 's' as the regex separator character so we can
    // have "s/<regex>/<subst>/" or "s|<regex>|<subst>|"
    const char separator_char = regex_sed[first_separator_char_pos];
    const size_t second_separator_char_pos =
        regex_sed.find(separator_char, first_separator_char_pos + 1);

    if (second_separator_char_pos == std::string::npos) {
      error.SetErrorStringWithFormat(
          "missing second '%c' separator char after '%.*s' in '%.*s'",
          separator_char,
          (int)(regex_sed.size() - first_separator_char_pos - 1),
          regex_sed.data() + (first_separator_char_pos + 1),
          (int)regex_sed.size(), regex_sed.data());
      return error;
    }

    const size_t third_separator_char_pos =
        regex_sed.find(separator_char, second_separator_char_pos + 1);

    if (third_separator_char_pos == std::string::npos) {
      error.SetErrorStringWithFormat(
          "missing third '%c' separator char after '%.*s' in '%.*s'",
          separator_char,
          (int)(regex_sed.size() - second_separator_char_pos - 1),
          regex_sed.data() + (second_separator_char_pos + 1),
          (int)regex_sed.size(), regex_sed.data());
      return error;
    }

    if (third_separator_char_pos != regex_sed_size - 1) {
      // Make sure that everything that follows the last regex separator char
      if (regex_sed.find_first_not_of("\t\n\v\f\r ",
                                      third_separator_char_pos + 1) !=
          std::string::npos) {
        error.SetErrorStringWithFormat(
            "extra data found after the '%.*s' regular expression substitution "
            "string: '%.*s'",
            (int)third_separator_char_pos + 1, regex_sed.data(),
            (int)(regex_sed.size() - third_separator_char_pos - 1),
            regex_sed.data() + (third_separator_char_pos + 1));
        return error;
      }
    } else if (first_separator_char_pos + 1 == second_separator_char_pos) {
      error.SetErrorStringWithFormat(
          "<regex> can't be empty in 's%c<regex>%c<subst>%c' string: '%.*s'",
          separator_char, separator_char, separator_char, (int)regex_sed.size(),
          regex_sed.data());
      return error;
    } else if (second_separator_char_pos + 1 == third_separator_char_pos) {
      error.SetErrorStringWithFormat(
          "<subst> can't be empty in 's%c<regex>%c<subst>%c' string: '%.*s'",
          separator_char, separator_char, separator_char, (int)regex_sed.size(),
          regex_sed.data());
      return error;
    }

    if (!check_only) {
      std::string regex(std::string(regex_sed.substr(
          first_separator_char_pos + 1,
          second_separator_char_pos - first_separator_char_pos - 1)));
      std::string subst(std::string(regex_sed.substr(
          second_separator_char_pos + 1,
          third_separator_char_pos - second_separator_char_pos - 1)));
      m_regex_cmd_up->AddRegexCommand(regex, subst);
    }
    return error;
  }

  void AddRegexCommandToInterpreter() {
    if (m_regex_cmd_up) {
      if (m_regex_cmd_up->HasRegexEntries()) {
        CommandObjectSP cmd_sp(m_regex_cmd_up.release());
        m_interpreter.AddCommand(cmd_sp->GetCommandName(), cmd_sp, true);
      }
    }
  }

private:
  std::unique_ptr<CommandObjectRegexCommand> m_regex_cmd_up;

  class CommandOptions : public Options {
  public:
    CommandOptions() = default;

    ~CommandOptions() override = default;

    Status SetOptionValue(uint32_t option_idx, llvm::StringRef option_arg,
                          ExecutionContext *execution_context) override {
      Status error;
      const int short_option = m_getopt_table[option_idx].val;

      switch (short_option) {
      case 'h':
        m_help.assign(std::string(option_arg));
        break;
      case 's':
        m_syntax.assign(std::string(option_arg));
        break;
      default:
        llvm_unreachable("Unimplemented option");
      }

      return error;
    }

    void OptionParsingStarting(ExecutionContext *execution_context) override {
      m_help.clear();
      m_syntax.clear();
    }

    llvm::ArrayRef<OptionDefinition> GetDefinitions() override {
      return llvm::makeArrayRef(g_regex_options);
    }

    llvm::StringRef GetHelp() { return m_help; }

    llvm::StringRef GetSyntax() { return m_syntax; }

  protected:
    // Instance variables to hold the values for command options.

    std::string m_help;
    std::string m_syntax;
  };

  Options *GetOptions() override { return &m_options; }

  CommandOptions m_options;
};

class CommandObjectPythonFunction : public CommandObjectRaw {
public:
  CommandObjectPythonFunction(CommandInterpreter &interpreter, std::string name,
                              std::string funct, std::string help,
                              ScriptedCommandSynchronicity synch)
      : CommandObjectRaw(interpreter, name), m_function_name(funct),
        m_synchro(synch) {
    if (!help.empty())
      SetHelp(help);
    else {
      StreamString stream;
      stream.Printf("For more information run 'help %s'", name.c_str());
      SetHelp(stream.GetString());
    }
  }

  ~CommandObjectPythonFunction() override = default;

  bool IsRemovable() const override { return true; }

  const std::string &GetFunctionName() { return m_function_name; }

  ScriptedCommandSynchronicity GetSynchronicity() { return m_synchro; }

  llvm::StringRef GetHelpLong() override {
    if (m_fetched_help_long)
      return CommandObjectRaw::GetHelpLong();

    ScriptInterpreter *scripter = GetDebugger().GetScriptInterpreter();
    if (!scripter)
      return CommandObjectRaw::GetHelpLong();

    std::string docstring;
    m_fetched_help_long =
        scripter->GetDocumentationForItem(m_function_name.c_str(), docstring);
    if (!docstring.empty())
      SetHelpLong(docstring);
    return CommandObjectRaw::GetHelpLong();
  }

protected:
  bool DoExecute(llvm::StringRef raw_command_line,
                 CommandReturnObject &result) override {
    ScriptInterpreter *scripter = GetDebugger().GetScriptInterpreter();

    Status error;

    result.SetStatus(eReturnStatusInvalid);

    if (!scripter || !scripter->RunScriptBasedCommand(
                         m_function_name.c_str(), raw_command_line, m_synchro,
                         result, error, m_exe_ctx)) {
      result.AppendError(error.AsCString());
    } else {
      // Don't change the status if the command already set it...
      if (result.GetStatus() == eReturnStatusInvalid) {
        if (result.GetOutputData().empty())
          result.SetStatus(eReturnStatusSuccessFinishNoResult);
        else
          result.SetStatus(eReturnStatusSuccessFinishResult);
      }
    }

    return result.Succeeded();
  }

private:
  std::string m_function_name;
  ScriptedCommandSynchronicity m_synchro;
  bool m_fetched_help_long = false;
};

class CommandObjectScriptingObject : public CommandObjectRaw {
public:
  CommandObjectScriptingObject(CommandInterpreter &interpreter,
                               std::string name,
                               StructuredData::GenericSP cmd_obj_sp,
                               ScriptedCommandSynchronicity synch)
      : CommandObjectRaw(interpreter, name), m_cmd_obj_sp(cmd_obj_sp),
        m_synchro(synch), m_fetched_help_short(false),
        m_fetched_help_long(false) {
    StreamString stream;
    stream.Printf("For more information run 'help %s'", name.c_str());
    SetHelp(stream.GetString());
    if (ScriptInterpreter *scripter = GetDebugger().GetScriptInterpreter())
      GetFlags().Set(scripter->GetFlagsForCommandObject(cmd_obj_sp));
  }

  ~CommandObjectScriptingObject() override = default;

  bool IsRemovable() const override { return true; }

  ScriptedCommandSynchronicity GetSynchronicity() { return m_synchro; }

  llvm::StringRef GetHelp() override {
    if (m_fetched_help_short)
      return CommandObjectRaw::GetHelp();
    ScriptInterpreter *scripter = GetDebugger().GetScriptInterpreter();
    if (!scripter)
      return CommandObjectRaw::GetHelp();
    std::string docstring;
    m_fetched_help_short =
        scripter->GetShortHelpForCommandObject(m_cmd_obj_sp, docstring);
    if (!docstring.empty())
      SetHelp(docstring);

    return CommandObjectRaw::GetHelp();
  }

  llvm::StringRef GetHelpLong() override {
    if (m_fetched_help_long)
      return CommandObjectRaw::GetHelpLong();

    ScriptInterpreter *scripter = GetDebugger().GetScriptInterpreter();
    if (!scripter)
      return CommandObjectRaw::GetHelpLong();

    std::string docstring;
    m_fetched_help_long =
        scripter->GetLongHelpForCommandObject(m_cmd_obj_sp, docstring);
    if (!docstring.empty())
      SetHelpLong(docstring);
    return CommandObjectRaw::GetHelpLong();
  }

protected:
  bool DoExecute(llvm::StringRef raw_command_line,
                 CommandReturnObject &result) override {
    ScriptInterpreter *scripter = GetDebugger().GetScriptInterpreter();

    Status error;

    result.SetStatus(eReturnStatusInvalid);

    if (!scripter ||
        !scripter->RunScriptBasedCommand(m_cmd_obj_sp, raw_command_line,
                                         m_synchro, result, error, m_exe_ctx)) {
      result.AppendError(error.AsCString());
    } else {
      // Don't change the status if the command already set it...
      if (result.GetStatus() == eReturnStatusInvalid) {
        if (result.GetOutputData().empty())
          result.SetStatus(eReturnStatusSuccessFinishNoResult);
        else
          result.SetStatus(eReturnStatusSuccessFinishResult);
      }
    }

    return result.Succeeded();
  }

private:
  StructuredData::GenericSP m_cmd_obj_sp;
  ScriptedCommandSynchronicity m_synchro;
  bool m_fetched_help_short : 1;
  bool m_fetched_help_long : 1;
};

// CommandObjectCommandsScriptImport
#define LLDB_OPTIONS_script_import
#include "CommandOptions.inc"

class CommandObjectCommandsScriptImport : public CommandObjectParsed {
public:
  CommandObjectCommandsScriptImport(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "command script import",
                            "Import a scripting module in LLDB.", nullptr) {
    CommandArgumentEntry arg1;
    CommandArgumentData cmd_arg;

    // Define the first (and only) variant of this arg.
    cmd_arg.arg_type = eArgTypeFilename;
    cmd_arg.arg_repetition = eArgRepeatPlus;

    // There is only one variant this argument could be; put it into the
    // argument entry.
    arg1.push_back(cmd_arg);

    // Push the data for the first argument into the m_arguments vector.
    m_arguments.push_back(arg1);
  }

  ~CommandObjectCommandsScriptImport() override = default;

  void
  HandleArgumentCompletion(CompletionRequest &request,
                           OptionElementVector &opt_element_vector) override {
    CommandCompletions::InvokeCommonCompletionCallbacks(
        GetCommandInterpreter(), CommandCompletions::eDiskFileCompletion,
        request, nullptr);
  }

  Options *GetOptions() override { return &m_options; }

protected:
  class CommandOptions : public Options {
  public:
    CommandOptions() = default;

    ~CommandOptions() override = default;

    Status SetOptionValue(uint32_t option_idx, llvm::StringRef option_arg,
                          ExecutionContext *execution_context) override {
      Status error;
      const int short_option = m_getopt_table[option_idx].val;

      switch (short_option) {
      case 'r':
        // NO-OP
        break;
      case 'c':
        relative_to_command_file = true;
        break;
      case 's':
        silent = true;
        break;
      default:
        llvm_unreachable("Unimplemented option");
      }

      return error;
    }

    void OptionParsingStarting(ExecutionContext *execution_context) override {
      relative_to_command_file = false;
    }

    llvm::ArrayRef<OptionDefinition> GetDefinitions() override {
      return llvm::makeArrayRef(g_script_import_options);
    }
    bool relative_to_command_file = false;
    bool silent = false;
  };

  bool DoExecute(Args &command, CommandReturnObject &result) override {
    if (command.empty()) {
      result.AppendError("command script import needs one or more arguments");
      return false;
    }

    FileSpec source_dir = {};
    if (m_options.relative_to_command_file) {
      source_dir = GetDebugger().GetCommandInterpreter().GetCurrentSourceDir();
      if (!source_dir) {
        result.AppendError("command script import -c can only be specified "
                           "from a command file");
        return false;
      }
    }

    for (auto &entry : command.entries()) {
      Status error;

      LoadScriptOptions options;
      options.SetInitSession(true);
      options.SetSilent(m_options.silent);

      // FIXME: this is necessary because CommandObject::CheckRequirements()
      // assumes that commands won't ever be recursively invoked, but it's
      // actually possible to craft a Python script that does other "command
      // script imports" in __lldb_init_module the real fix is to have
      // recursive commands possible with a CommandInvocation object separate
      // from the CommandObject itself, so that recursive command invocations
      // won't stomp on each other (wrt to execution contents, options, and
      // more)
      m_exe_ctx.Clear();
      if (GetDebugger().GetScriptInterpreter()->LoadScriptingModule(
              entry.c_str(), options, error, /*module_sp=*/nullptr,
              source_dir)) {
        result.SetStatus(eReturnStatusSuccessFinishNoResult);
      } else {
        result.AppendErrorWithFormat("module importing failed: %s",
                                     error.AsCString());
      }
    }

    return result.Succeeded();
  }

  CommandOptions m_options;
};

#define LLDB_OPTIONS_script_add
#include "CommandOptions.inc"

class CommandObjectCommandsScriptAdd : public CommandObjectParsed,
                                       public IOHandlerDelegateMultiline {
public:
  CommandObjectCommandsScriptAdd(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "command script add",
                            "Add a scripted function as an LLDB command.",
                            "Add a scripted function as an lldb command. "
                            "If you provide a single argument, the command "
                            "will be added at the root level of the command "
                            "hierarchy.  If there are more arguments they "
                            "must be a path to a user-added container "
                            "command, and the last element will be the new "
                            "command name."),
        IOHandlerDelegateMultiline("DONE") {
    CommandArgumentEntry arg1;
    CommandArgumentData cmd_arg;

    // This is one or more command names, which form the path to the command
    // you want to add.
    cmd_arg.arg_type = eArgTypeCommand;
    cmd_arg.arg_repetition = eArgRepeatPlus;

    // There is only one variant this argument could be; put it into the
    // argument entry.
    arg1.push_back(cmd_arg);

    // Push the data for the first argument into the m_arguments vector.
    m_arguments.push_back(arg1);
  }

  ~CommandObjectCommandsScriptAdd() override = default;

  Options *GetOptions() override { return &m_options; }

  void
  HandleArgumentCompletion(CompletionRequest &request,
                           OptionElementVector &opt_element_vector) override {
    CommandCompletions::CompleteModifiableCmdPathArgs(m_interpreter, request,
                                                      opt_element_vector);
  }

protected:
  class CommandOptions : public Options {
  public:
    CommandOptions() = default;

    ~CommandOptions() override = default;

    Status SetOptionValue(uint32_t option_idx, llvm::StringRef option_arg,
                          ExecutionContext *execution_context) override {
      Status error;
      const int short_option = m_getopt_table[option_idx].val;

      switch (short_option) {
      case 'f':
        if (!option_arg.empty())
          m_funct_name = std::string(option_arg);
        break;
      case 'c':
        if (!option_arg.empty())
          m_class_name = std::string(option_arg);
        break;
      case 'h':
        if (!option_arg.empty())
          m_short_help = std::string(option_arg);
        break;
      case 'o':
        m_overwrite_lazy = eLazyBoolYes;
        break;
      case 's':
        m_synchronicity =
            (ScriptedCommandSynchronicity)OptionArgParser::ToOptionEnum(
                option_arg, GetDefinitions()[option_idx].enum_values, 0, error);
        if (!error.Success())
          error.SetErrorStringWithFormat(
              "unrecognized value for synchronicity '%s'",
              option_arg.str().c_str());
        break;
      default:
        llvm_unreachable("Unimplemented option");
      }

      return error;
    }

    void OptionParsingStarting(ExecutionContext *execution_context) override {
      m_class_name.clear();
      m_funct_name.clear();
      m_short_help.clear();
      m_overwrite_lazy = eLazyBoolCalculate;
      m_synchronicity = eScriptedCommandSynchronicitySynchronous;
    }

    llvm::ArrayRef<OptionDefinition> GetDefinitions() override {
      return llvm::makeArrayRef(g_script_add_options);
    }

    // Instance variables to hold the values for command options.

    std::string m_class_name;
    std::string m_funct_name;
    std::string m_short_help;
    LazyBool m_overwrite_lazy = eLazyBoolCalculate;
    ScriptedCommandSynchronicity m_synchronicity =
        eScriptedCommandSynchronicitySynchronous;
  };

  void IOHandlerActivated(IOHandler &io_handler, bool interactive) override {
    StreamFileSP output_sp(io_handler.GetOutputStreamFileSP());
    if (output_sp && interactive) {
      output_sp->PutCString(g_python_command_instructions);
      output_sp->Flush();
    }
  }

  void IOHandlerInputComplete(IOHandler &io_handler,
                              std::string &data) override {
    StreamFileSP error_sp = io_handler.GetErrorStreamFileSP();

    ScriptInterpreter *interpreter = GetDebugger().GetScriptInterpreter();
    if (interpreter) {
      StringList lines;
      lines.SplitIntoLines(data);
      if (lines.GetSize() > 0) {
        std::string funct_name_str;
        if (interpreter->GenerateScriptAliasFunction(lines, funct_name_str)) {
          if (funct_name_str.empty()) {
            error_sp->Printf("error: unable to obtain a function name, didn't "
                             "add python command.\n");
            error_sp->Flush();
          } else {
            // everything should be fine now, let's add this alias

            CommandObjectSP command_obj_sp(new CommandObjectPythonFunction(
                m_interpreter, m_cmd_name, funct_name_str, m_short_help,
                m_synchronicity));
            if (!m_container) {
              Status error = m_interpreter.AddUserCommand(
                  m_cmd_name, command_obj_sp, m_overwrite);
              if (error.Fail()) {
                error_sp->Printf("error: unable to add selected command: '%s'",
                                 error.AsCString());
                error_sp->Flush();
              }
            } else {
              llvm::Error llvm_error = m_container->LoadUserSubcommand(
                  m_cmd_name, command_obj_sp, m_overwrite);
              if (llvm_error) {
                error_sp->Printf("error: unable to add selected command: '%s'",
                               llvm::toString(std::move(llvm_error)).c_str());
                error_sp->Flush();
              }
            }
          }
        } else {
          error_sp->Printf(
              "error: unable to create function, didn't add python command\n");
          error_sp->Flush();
        }
      } else {
        error_sp->Printf("error: empty function, didn't add python command\n");
        error_sp->Flush();
      }
    } else {
      error_sp->Printf(
          "error: script interpreter missing, didn't add python command\n");
      error_sp->Flush();
    }

    io_handler.SetIsDone(true);
  }

  bool DoExecute(Args &command, CommandReturnObject &result) override {
    if (GetDebugger().GetScriptLanguage() != lldb::eScriptLanguagePython) {
      result.AppendError("only scripting language supported for scripted "
                         "commands is currently Python");
      return false;
    }

    if (command.GetArgumentCount() == 0) {
      result.AppendError("'command script add' requires at least one argument");
      return false;
    }
    // Store the options in case we get multi-line input, also figure out the
    // default if not user supplied:
    switch (m_options.m_overwrite_lazy) {
      case eLazyBoolCalculate:
        m_overwrite = !GetDebugger().GetCommandInterpreter().GetRequireCommandOverwrite();
        break;
      case eLazyBoolYes:
        m_overwrite = true;
        break;
      case eLazyBoolNo:
        m_overwrite = false;
    }
    
    Status path_error;
    m_container = GetCommandInterpreter().VerifyUserMultiwordCmdPath(
        command, true, path_error);

    if (path_error.Fail()) {
      result.AppendErrorWithFormat("error in command path: %s",
                                   path_error.AsCString());
      return false;
    }

    if (!m_container) {
      // This is getting inserted into the root of the interpreter.
      m_cmd_name = std::string(command[0].ref());
    } else {
      size_t num_args = command.GetArgumentCount();
      m_cmd_name = std::string(command[num_args - 1].ref());
    }

    m_short_help.assign(m_options.m_short_help);
    m_synchronicity = m_options.m_synchronicity;

    // Handle the case where we prompt for the script code first:
    if (m_options.m_class_name.empty() && m_options.m_funct_name.empty()) {
      m_interpreter.GetPythonCommandsFromIOHandler("     ", // Prompt
                                                   *this);  // IOHandlerDelegate
      return result.Succeeded();
    }

    CommandObjectSP new_cmd_sp;
    if (m_options.m_class_name.empty()) {
      new_cmd_sp.reset(new CommandObjectPythonFunction(
          m_interpreter, m_cmd_name, m_options.m_funct_name,
          m_options.m_short_help, m_synchronicity));
    } else {
      ScriptInterpreter *interpreter = GetDebugger().GetScriptInterpreter();
      if (!interpreter) {
        result.AppendError("cannot find ScriptInterpreter");
        return false;
      }

      auto cmd_obj_sp = interpreter->CreateScriptCommandObject(
          m_options.m_class_name.c_str());
      if (!cmd_obj_sp) {
        result.AppendError("cannot create helper object");
        return false;
      }

      new_cmd_sp.reset(new CommandObjectScriptingObject(
          m_interpreter, m_cmd_name, cmd_obj_sp, m_synchronicity));
    }
    
    // Assume we're going to succeed...
    result.SetStatus(eReturnStatusSuccessFinishNoResult);
    if (!m_container) {
      Status add_error =
          m_interpreter.AddUserCommand(m_cmd_name, new_cmd_sp, m_overwrite);
      if (add_error.Fail())
        result.AppendErrorWithFormat("cannot add command: %s",
                                     add_error.AsCString());
    } else {
      llvm::Error llvm_error =
          m_container->LoadUserSubcommand(m_cmd_name, new_cmd_sp, m_overwrite);
      if (llvm_error)
        result.AppendErrorWithFormat("cannot add command: %s", 
                                     llvm::toString(std::move(llvm_error)).c_str());
    }
    return result.Succeeded();
  }

  CommandOptions m_options;
  std::string m_cmd_name;
  CommandObjectMultiword *m_container = nullptr;
  std::string m_short_help;
  bool m_overwrite = false;
  ScriptedCommandSynchronicity m_synchronicity =
      eScriptedCommandSynchronicitySynchronous;
};

// CommandObjectCommandsScriptList

class CommandObjectCommandsScriptList : public CommandObjectParsed {
public:
  CommandObjectCommandsScriptList(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "command script list",
                            "List defined top-level scripted commands.",
                            nullptr) {}

  ~CommandObjectCommandsScriptList() override = default;

  bool DoExecute(Args &command, CommandReturnObject &result) override {
    m_interpreter.GetHelp(result, CommandInterpreter::eCommandTypesUserDef);

    result.SetStatus(eReturnStatusSuccessFinishResult);

    return true;
  }
};

// CommandObjectCommandsScriptClear

class CommandObjectCommandsScriptClear : public CommandObjectParsed {
public:
  CommandObjectCommandsScriptClear(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "command script clear",
                            "Delete all scripted commands.", nullptr) {}

  ~CommandObjectCommandsScriptClear() override = default;

protected:
  bool DoExecute(Args &command, CommandReturnObject &result) override {
    m_interpreter.RemoveAllUser();

    result.SetStatus(eReturnStatusSuccessFinishResult);

    return true;
  }
};

// CommandObjectCommandsScriptDelete

class CommandObjectCommandsScriptDelete : public CommandObjectParsed {
public:
  CommandObjectCommandsScriptDelete(CommandInterpreter &interpreter)
      : CommandObjectParsed(
            interpreter, "command script delete",
            "Delete a scripted command by specifying the path to the command.",
            nullptr) {
    CommandArgumentEntry arg1;
    CommandArgumentData cmd_arg;

    // This is a list of command names forming the path to the command
    // to be deleted.
    cmd_arg.arg_type = eArgTypeCommand;
    cmd_arg.arg_repetition = eArgRepeatPlus;

    // There is only one variant this argument could be; put it into the
    // argument entry.
    arg1.push_back(cmd_arg);

    // Push the data for the first argument into the m_arguments vector.
    m_arguments.push_back(arg1);
  }

  ~CommandObjectCommandsScriptDelete() override = default;

  void
  HandleArgumentCompletion(CompletionRequest &request,
                           OptionElementVector &opt_element_vector) override {
    CommandCompletions::CompleteModifiableCmdPathArgs(m_interpreter, request,
                                                      opt_element_vector);
  }

protected:
  bool DoExecute(Args &command, CommandReturnObject &result) override {

    llvm::StringRef root_cmd = command[0].ref();
    size_t num_args = command.GetArgumentCount();

    if (root_cmd.empty()) {
      result.AppendErrorWithFormat("empty root command name");
      return false;
    }
    if (!m_interpreter.HasUserCommands() &&
        !m_interpreter.HasUserMultiwordCommands()) {
      result.AppendErrorWithFormat("can only delete user defined commands, "
                                   "but no user defined commands found");
      return false;
    }

    CommandObjectSP cmd_sp = m_interpreter.GetCommandSPExact(root_cmd);
    if (!cmd_sp) {
      result.AppendErrorWithFormat("command '%s' not found.",
                                   command[0].c_str());
      return false;
    }
    if (!cmd_sp->IsUserCommand()) {
      result.AppendErrorWithFormat("command '%s' is not a user command.",
                                   command[0].c_str());
      return false;
    }
    if (cmd_sp->GetAsMultiwordCommand() && num_args == 1) {
      result.AppendErrorWithFormat("command '%s' is a multi-word command.\n "
                                   "Delete with \"command container delete\"",
                                   command[0].c_str());
      return false;
    }

    if (command.GetArgumentCount() == 1) {
      m_interpreter.RemoveUser(root_cmd);
      result.SetStatus(eReturnStatusSuccessFinishResult);
      return true;
    }
    // We're deleting a command from a multiword command.  Verify the command
    // path:
    Status error;
    CommandObjectMultiword *container =
        GetCommandInterpreter().VerifyUserMultiwordCmdPath(command, true,
                                                           error);
    if (error.Fail()) {
      result.AppendErrorWithFormat("could not resolve command path: %s",
                                   error.AsCString());
      return false;
    }
    if (!container) {
      // This means that command only had a leaf command, so the container is
      // the root.  That should have been handled above.
      result.AppendErrorWithFormat("could not find a container for '%s'",
                                   command[0].c_str());
      return false;
    }
    const char *leaf_cmd = command[num_args - 1].c_str();
    llvm::Error llvm_error = container->RemoveUserSubcommand(leaf_cmd,
                                            /* multiword not okay */ false);
    if (llvm_error) {
      result.AppendErrorWithFormat("could not delete command '%s': %s",
                                   leaf_cmd, 
                                   llvm::toString(std::move(llvm_error)).c_str());
      return false;
    }

    Stream &out_stream = result.GetOutputStream();

    out_stream << "Deleted command:";
    for (size_t idx = 0; idx < num_args; idx++) {
      out_stream << ' ';
      out_stream << command[idx].c_str();
    }
    out_stream << '\n';
    result.SetStatus(eReturnStatusSuccessFinishResult);
    return true;
  }
};

#pragma mark CommandObjectMultiwordCommandsScript

// CommandObjectMultiwordCommandsScript

class CommandObjectMultiwordCommandsScript : public CommandObjectMultiword {
public:
  CommandObjectMultiwordCommandsScript(CommandInterpreter &interpreter)
      : CommandObjectMultiword(
            interpreter, "command script",
            "Commands for managing custom "
            "commands implemented by "
            "interpreter scripts.",
            "command script <subcommand> [<subcommand-options>]") {
    LoadSubCommand("add", CommandObjectSP(
                              new CommandObjectCommandsScriptAdd(interpreter)));
    LoadSubCommand(
        "delete",
        CommandObjectSP(new CommandObjectCommandsScriptDelete(interpreter)));
    LoadSubCommand(
        "clear",
        CommandObjectSP(new CommandObjectCommandsScriptClear(interpreter)));
    LoadSubCommand("list", CommandObjectSP(new CommandObjectCommandsScriptList(
                               interpreter)));
    LoadSubCommand(
        "import",
        CommandObjectSP(new CommandObjectCommandsScriptImport(interpreter)));
  }

  ~CommandObjectMultiwordCommandsScript() override = default;
};

#pragma mark CommandObjectCommandContainer
#define LLDB_OPTIONS_container_add
#include "CommandOptions.inc"

class CommandObjectCommandsContainerAdd : public CommandObjectParsed {
public:
  CommandObjectCommandsContainerAdd(CommandInterpreter &interpreter)
      : CommandObjectParsed(
            interpreter, "command container add",
            "Add a container command to lldb.  Adding to built-"
            "in container commands is not allowed.",
            "command container add [[path1]...] container-name") {
    CommandArgumentEntry arg1;
    CommandArgumentData cmd_arg;

    // This is one or more command names, which form the path to the command
    // you want to add.
    cmd_arg.arg_type = eArgTypeCommand;
    cmd_arg.arg_repetition = eArgRepeatPlus;

    // There is only one variant this argument could be; put it into the
    // argument entry.
    arg1.push_back(cmd_arg);

    // Push the data for the first argument into the m_arguments vector.
    m_arguments.push_back(arg1);
  }

  ~CommandObjectCommandsContainerAdd() override = default;

  Options *GetOptions() override { return &m_options; }

  void
  HandleArgumentCompletion(CompletionRequest &request,
                           OptionElementVector &opt_element_vector) override {
    CommandCompletions::CompleteModifiableCmdPathArgs(m_interpreter, request,
                                                      opt_element_vector);
  }

protected:
  class CommandOptions : public Options {
  public:
    CommandOptions() = default;

    ~CommandOptions() override = default;

    Status SetOptionValue(uint32_t option_idx, llvm::StringRef option_arg,
                          ExecutionContext *execution_context) override {
      Status error;
      const int short_option = m_getopt_table[option_idx].val;

      switch (short_option) {
      case 'h':
        if (!option_arg.empty())
          m_short_help = std::string(option_arg);
        break;
      case 'o':
        m_overwrite = true;
        break;
      case 'H':
        if (!option_arg.empty())
          m_long_help = std::string(option_arg);
        break;
      default:
        llvm_unreachable("Unimplemented option");
      }

      return error;
    }

    void OptionParsingStarting(ExecutionContext *execution_context) override {
      m_short_help.clear();
      m_long_help.clear();
      m_overwrite = false;
    }

    llvm::ArrayRef<OptionDefinition> GetDefinitions() override {
      return llvm::makeArrayRef(g_container_add_options);
    }

    // Instance variables to hold the values for command options.

    std::string m_short_help;
    std::string m_long_help;
    bool m_overwrite = false;
  };
  bool DoExecute(Args &command, CommandReturnObject &result) override {
    size_t num_args = command.GetArgumentCount();

    if (num_args == 0) {
      result.AppendError("no command was specified");
      return false;
    }

    if (num_args == 1) {
      // We're adding this as a root command, so use the interpreter.
      const char *cmd_name = command.GetArgumentAtIndex(0);
      auto cmd_sp = CommandObjectSP(new CommandObjectMultiword(
          GetCommandInterpreter(), cmd_name, m_options.m_short_help.c_str(),
          m_options.m_long_help.c_str()));
      cmd_sp->GetAsMultiwordCommand()->SetRemovable(true);
      Status add_error = GetCommandInterpreter().AddUserCommand(
          cmd_name, cmd_sp, m_options.m_overwrite);
      if (add_error.Fail()) {
        result.AppendErrorWithFormat("error adding command: %s",
                                     add_error.AsCString());
        return false;
      }
      result.SetStatus(eReturnStatusSuccessFinishNoResult);
      return true;
    }

    // We're adding this to a subcommand, first find the subcommand:
    Status path_error;
    CommandObjectMultiword *add_to_me =
        GetCommandInterpreter().VerifyUserMultiwordCmdPath(command, true,
                                                           path_error);

    if (!add_to_me) {
      result.AppendErrorWithFormat("error adding command: %s",
                                   path_error.AsCString());
      return false;
    }

    const char *cmd_name = command.GetArgumentAtIndex(num_args - 1);
    auto cmd_sp = CommandObjectSP(new CommandObjectMultiword(
        GetCommandInterpreter(), cmd_name, m_options.m_short_help.c_str(),
        m_options.m_long_help.c_str()));
    llvm::Error llvm_error =
        add_to_me->LoadUserSubcommand(cmd_name, cmd_sp, m_options.m_overwrite);
    if (llvm_error) {
      result.AppendErrorWithFormat("error adding subcommand: %s",
                                   llvm::toString(std::move(llvm_error)).c_str());
      return false;
    }

    result.SetStatus(eReturnStatusSuccessFinishNoResult);
    return true;
  }

private:
  CommandOptions m_options;
};

#define LLDB_OPTIONS_multiword_delete
#include "CommandOptions.inc"
class CommandObjectCommandsContainerDelete : public CommandObjectParsed {
public:
  CommandObjectCommandsContainerDelete(CommandInterpreter &interpreter)
      : CommandObjectParsed(
            interpreter, "command container delete",
            "Delete a container command previously added to "
            "lldb.",
            "command container delete [[path1] ...] container-cmd") {
    CommandArgumentEntry arg1;
    CommandArgumentData cmd_arg;

    // This is one or more command names, which form the path to the command
    // you want to add.
    cmd_arg.arg_type = eArgTypeCommand;
    cmd_arg.arg_repetition = eArgRepeatPlus;

    // There is only one variant this argument could be; put it into the
    // argument entry.
    arg1.push_back(cmd_arg);

    // Push the data for the first argument into the m_arguments vector.
    m_arguments.push_back(arg1);
  }

  ~CommandObjectCommandsContainerDelete() override = default;

  void
  HandleArgumentCompletion(CompletionRequest &request,
                           OptionElementVector &opt_element_vector) override {
    CommandCompletions::CompleteModifiableCmdPathArgs(m_interpreter, request,
                                                      opt_element_vector);
  }

protected:
  bool DoExecute(Args &command, CommandReturnObject &result) override {
    size_t num_args = command.GetArgumentCount();

    if (num_args == 0) {
      result.AppendError("No command was specified.");
      return false;
    }

    if (num_args == 1) {
      // We're removing a root command, so we need to delete it from the
      // interpreter.
      const char *cmd_name = command.GetArgumentAtIndex(0);
      // Let's do a little more work here so we can do better error reporting.
      CommandInterpreter &interp = GetCommandInterpreter();
      CommandObjectSP cmd_sp = interp.GetCommandSPExact(cmd_name);
      if (!cmd_sp) {
        result.AppendErrorWithFormat("container command %s doesn't exist.",
                                     cmd_name);
        return false;
      }
      if (!cmd_sp->IsUserCommand()) {
        result.AppendErrorWithFormat(
            "container command %s is not a user command", cmd_name);
        return false;
      }
      if (!cmd_sp->GetAsMultiwordCommand()) {
        result.AppendErrorWithFormat("command %s is not a container command",
                                     cmd_name);
        return false;
      }

      bool did_remove = GetCommandInterpreter().RemoveUserMultiword(cmd_name);
      if (!did_remove) {
        result.AppendErrorWithFormat("error removing command %s.", cmd_name);
        return false;
      }

      result.SetStatus(eReturnStatusSuccessFinishNoResult);
      return true;
    }

    // We're removing a subcommand, first find the subcommand's owner:
    Status path_error;
    CommandObjectMultiword *container =
        GetCommandInterpreter().VerifyUserMultiwordCmdPath(command, true,
                                                           path_error);

    if (!container) {
      result.AppendErrorWithFormat("error removing container command: %s",
                                   path_error.AsCString());
      return false;
    }
    const char *leaf = command.GetArgumentAtIndex(num_args - 1);
    llvm::Error llvm_error =
        container->RemoveUserSubcommand(leaf, /* multiword okay */ true);
    if (llvm_error) {
      result.AppendErrorWithFormat("error removing container command: %s",
                                   llvm::toString(std::move(llvm_error)).c_str());
      return false;
    }
    result.SetStatus(eReturnStatusSuccessFinishNoResult);
    return true;
  }
};

class CommandObjectCommandContainer : public CommandObjectMultiword {
public:
  CommandObjectCommandContainer(CommandInterpreter &interpreter)
      : CommandObjectMultiword(
            interpreter, "command container",
            "Commands for adding container commands to lldb.  "
            "Container commands are containers for other commands.  You can "
            "add nested container commands by specifying a command path, "
            "but you can't add commands into the built-in command hierarchy.",
            "command container <subcommand> [<subcommand-options>]") {
    LoadSubCommand("add", CommandObjectSP(new CommandObjectCommandsContainerAdd(
                              interpreter)));
    LoadSubCommand(
        "delete",
        CommandObjectSP(new CommandObjectCommandsContainerDelete(interpreter)));
  }

  ~CommandObjectCommandContainer() override = default;
};

#pragma mark CommandObjectMultiwordCommands

// CommandObjectMultiwordCommands

CommandObjectMultiwordCommands::CommandObjectMultiwordCommands(
    CommandInterpreter &interpreter)
    : CommandObjectMultiword(interpreter, "command",
                             "Commands for managing custom LLDB commands.",
                             "command <subcommand> [<subcommand-options>]") {
  LoadSubCommand("source",
                 CommandObjectSP(new CommandObjectCommandsSource(interpreter)));
  LoadSubCommand("alias",
                 CommandObjectSP(new CommandObjectCommandsAlias(interpreter)));
  LoadSubCommand("unalias", CommandObjectSP(
                                new CommandObjectCommandsUnalias(interpreter)));
  LoadSubCommand("delete",
                 CommandObjectSP(new CommandObjectCommandsDelete(interpreter)));
  LoadSubCommand("container", CommandObjectSP(new CommandObjectCommandContainer(
                                  interpreter)));
  LoadSubCommand(
      "regex", CommandObjectSP(new CommandObjectCommandsAddRegex(interpreter)));
  LoadSubCommand(
      "script",
      CommandObjectSP(new CommandObjectMultiwordCommandsScript(interpreter)));
}

CommandObjectMultiwordCommands::~CommandObjectMultiwordCommands() = default;
