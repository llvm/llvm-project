//===-- CommandObjectHelp.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "CommandObjectHelp.h"
#include "lldb/Interpreter/CommandObjectMultiword.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Interpreter/CommandReturnObject.h"

using namespace lldb;
using namespace lldb_private;

//-------------------------------------------------------------------------
// CommandObjectHelp
//-------------------------------------------------------------------------

void
CommandObjectHelp::GenerateAdditionalHelpAvenuesMessage (Stream *s,
                                                         const char* command,
                                                         const char* prefix,
                                                         const char* subcommand,
                                                         bool include_apropos,
                                                         bool include_type_lookup)
{
    if (s && command && *command)
    {
        s->Printf("'%s' is not a known command.\n", command);
        s->Printf("Try '%shelp' to see a current list of commands.\n", prefix ? prefix : "");
        if (include_apropos)
        {
	  s->Printf("Try '%sapropos %s' for a list of related commands.\n",
                    prefix ? prefix : "", subcommand ? subcommand : command);
        }
        if (include_type_lookup)
        {
            s->Printf("Try '%stype lookup %s' for information on types, methods, functions, modules, etc.",
		      prefix ? prefix : "", subcommand ? subcommand : command);
        }
    }
}

CommandObjectHelp::CommandObjectHelp(CommandInterpreter &interpreter)
    : CommandObjectParsed(interpreter, "help",
                          "Show a list of all debugger commands, or give details about a specific command.",
                          "help [<cmd-name>]"),
      m_options()
{
    CommandArgumentEntry arg;
    CommandArgumentData command_arg;

    // Define the first (and only) variant of this arg.
    command_arg.arg_type = eArgTypeCommandName;
    command_arg.arg_repetition = eArgRepeatStar;

    // There is only one variant this argument could be; put it into the argument entry.
    arg.push_back (command_arg);

    // Push the data for the first argument into the m_arguments vector.
    m_arguments.push_back (arg);
}

CommandObjectHelp::~CommandObjectHelp() = default;

OptionDefinition
CommandObjectHelp::CommandOptions::g_option_table[] =
{
  // clang-format off
  {LLDB_OPT_SET_ALL, false, "hide-aliases",         'a', OptionParser::eNoArgument, nullptr, nullptr, 0, eArgTypeNone, "Hide aliases in the command list."},
  {LLDB_OPT_SET_ALL, false, "hide-user-commands",   'u', OptionParser::eNoArgument, nullptr, nullptr, 0, eArgTypeNone, "Hide user-defined commands from the list."},
  {LLDB_OPT_SET_ALL, false, "show-hidden-commands", 'h', OptionParser::eNoArgument, nullptr, nullptr, 0, eArgTypeNone, "Include commands prefixed with an underscore."},
  {0, false, nullptr, 0, 0, nullptr, nullptr, 0, eArgTypeNone, nullptr}
  // clang-format on
};

bool
CommandObjectHelp::DoExecute (Args& command, CommandReturnObject &result)
{
    CommandObject::CommandMap::iterator pos;
    CommandObject *cmd_obj;
    const size_t argc = command.GetArgumentCount ();
    
    // 'help' doesn't take any arguments, other than command names.  If argc is 0, we show the user
    // all commands (aliases and user commands if asked for).  Otherwise every argument must be the name of a command or a sub-command.
    if (argc == 0)
    {
        uint32_t cmd_types = CommandInterpreter::eCommandTypesBuiltin;
        if (m_options.m_show_aliases)
            cmd_types |= CommandInterpreter::eCommandTypesAliases;
        if (m_options.m_show_user_defined)
            cmd_types |= CommandInterpreter::eCommandTypesUserDef;
        if (m_options.m_show_hidden)
            cmd_types |= CommandInterpreter::eCommandTypesHidden;

        result.SetStatus (eReturnStatusSuccessFinishNoResult);
        m_interpreter.GetHelp (result, cmd_types);  // General help
    }
    else
    {
        // Get command object for the first command argument. Only search built-in command dictionary.
        StringList matches;
        cmd_obj = m_interpreter.GetCommandObject (command.GetArgumentAtIndex (0), &matches);
        bool is_alias_command = m_interpreter.AliasExists (command.GetArgumentAtIndex (0));
        std::string alias_name = command.GetArgumentAtIndex(0);
        
        if (cmd_obj != nullptr)
        {
            StringList matches;
            bool all_okay = true;
            CommandObject *sub_cmd_obj = cmd_obj;
            // Loop down through sub_command dictionaries until we find the command object that corresponds
            // to the help command entered.
            std::string sub_command;
            for (size_t i = 1; i < argc && all_okay; ++i)
            {
                sub_command = command.GetArgumentAtIndex(i);
                matches.Clear();
                if (sub_cmd_obj->IsAlias())
                    sub_cmd_obj = ((CommandAlias*)sub_cmd_obj)->GetUnderlyingCommand().get();
                if (! sub_cmd_obj->IsMultiwordObject ())
                {
                    all_okay = false;
                }
                else
                {
                    CommandObject *found_cmd;
                    found_cmd = sub_cmd_obj->GetSubcommandObject(sub_command.c_str(), &matches);
                    if (found_cmd == nullptr)
                        all_okay = false;
                    else if (matches.GetSize() > 1)
                        all_okay = false;
                    else
                        sub_cmd_obj = found_cmd;
                }
            }
            
            if (!all_okay || (sub_cmd_obj == nullptr))
            {
                std::string cmd_string;
                command.GetCommandString (cmd_string);
                if (matches.GetSize() >= 2)
                {
                    StreamString s;
                    s.Printf ("ambiguous command %s", cmd_string.c_str());
                    size_t num_matches = matches.GetSize();
                    for (size_t match_idx = 0; match_idx < num_matches; match_idx++)
                    {
                        s.Printf ("\n\t%s", matches.GetStringAtIndex(match_idx));
                    }
                    s.Printf ("\n");
                    result.AppendError(s.GetData());
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
                else if (!sub_cmd_obj)
                {
                    StreamString error_msg_stream;
                    GenerateAdditionalHelpAvenuesMessage(&error_msg_stream,
                                                         cmd_string.c_str(),
                                                         m_interpreter.GetCommandPrefix(),
                                                         sub_command.c_str());
                    result.AppendErrorWithFormat("%s",error_msg_stream.GetData());
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
                else
                {
                    GenerateAdditionalHelpAvenuesMessage(&result.GetOutputStream(),
                                                         cmd_string.c_str(),
                                                         m_interpreter.GetCommandPrefix(),
                                                         sub_command.c_str());
                    result.GetOutputStream().Printf("\nThe closest match is '%s'. Help on it follows.\n\n", sub_cmd_obj->GetCommandName());
                }
            }
            
            sub_cmd_obj->GenerateHelpText(result);
            
            if (is_alias_command)
            {
                StreamString sstr;
                m_interpreter.GetAlias(alias_name.c_str())->GetAliasExpansion(sstr);
                result.GetOutputStream().Printf ("\n'%s' is an abbreviation for %s\n", alias_name.c_str(), sstr.GetData());
            }
        }
        else if (matches.GetSize() > 0)
        {
            Stream &output_strm = result.GetOutputStream();
            output_strm.Printf("Help requested with ambiguous command name, possible completions:\n");
            const size_t match_count = matches.GetSize();
            for (size_t i = 0; i < match_count; i++)
            {
                output_strm.Printf("\t%s\n", matches.GetStringAtIndex(i));
            }
        }
        else
        {
            // Maybe the user is asking for help about a command argument rather than a command.
            const CommandArgumentType arg_type = CommandObject::LookupArgumentName (command.GetArgumentAtIndex (0));
            if (arg_type != eArgTypeLastArg)
            {
                Stream &output_strm = result.GetOutputStream ();
                CommandObject::GetArgumentHelp (output_strm, arg_type, m_interpreter);
                result.SetStatus (eReturnStatusSuccessFinishNoResult);
            }
            else
            {
                StreamString error_msg_stream;
                GenerateAdditionalHelpAvenuesMessage(&error_msg_stream, command.GetArgumentAtIndex(0), m_interpreter.GetCommandPrefix());
                result.AppendErrorWithFormat("%s",error_msg_stream.GetData());
                result.SetStatus (eReturnStatusFailed);
            }
        }
    }
    
    return result.Succeeded();
}

int
CommandObjectHelp::HandleCompletion(Args &input,
                                    int &cursor_index,
                                    int &cursor_char_position,
                                    int match_start_point,
                                    int max_return_elements,
                                    bool &word_complete,
                                    StringList &matches)
{
    // Return the completions of the commands in the help system:
    if (cursor_index == 0)
    {
        return m_interpreter.HandleCompletionMatches (input, 
                                                    cursor_index, 
                                                    cursor_char_position, 
                                                    match_start_point, 
                                                    max_return_elements, 
                                                    word_complete, 
                                                    matches);
    }
    else
    {
        CommandObject *cmd_obj = m_interpreter.GetCommandObject (input.GetArgumentAtIndex(0));
        
        // The command that they are getting help on might be ambiguous, in which case we should complete that,
        // otherwise complete with the command the user is getting help on...
        
        if (cmd_obj)
        {
            input.Shift();
            cursor_index--;
            return cmd_obj->HandleCompletion (input, 
                                              cursor_index, 
                                              cursor_char_position, 
                                              match_start_point, 
                                              max_return_elements, 
                                              word_complete, 
                                              matches);
        }
        else
        {
            return m_interpreter.HandleCompletionMatches (input, 
                                                        cursor_index, 
                                                        cursor_char_position, 
                                                        match_start_point, 
                                                        max_return_elements, 
                                                        word_complete, 
                                                        matches);
        }
    }
}
