//===-- CommandObjectHelp.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectHelp.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandObjectMultiword.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Interpreter/CommandReturnObject.h"

using namespace lldb;
using namespace lldb_private;

//-------------------------------------------------------------------------
// CommandObjectHelp
//-------------------------------------------------------------------------

CommandObjectHelp::CommandObjectHelp (CommandInterpreter &interpreter) :
    CommandObjectParsed (interpreter,
                         "help",
                         "Show a list of all debugger commands, or give details about specific commands.",
                         "help [<cmd-name>]"), m_options (interpreter)
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

CommandObjectHelp::~CommandObjectHelp()
{
}

OptionDefinition
CommandObjectHelp::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_ALL, false, "show-aliases", 'a', no_argument, NULL, 0, eArgTypeNone,         "Show aliases in the command list."},
    { LLDB_OPT_SET_ALL, false, "hide-user-commands", 'u', no_argument, NULL, 0, eArgTypeNone,         "Hide user-defined commands from the list."},
    { 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};

bool
CommandObjectHelp::DoExecute (Args& command, CommandReturnObject &result)
{
    CommandObject::CommandMap::iterator pos;
    CommandObject *cmd_obj;
    const int argc = command.GetArgumentCount ();
    
    // 'help' doesn't take any arguments, other than command names.  If argc is 0, we show the user
    // all commands (aliases and user commands if asked for).  Otherwise every argument must be the name of a command or a sub-command.
    if (argc == 0)
    {
        uint32_t cmd_types = CommandInterpreter::eCommandTypesBuiltin;
        if (m_options.m_show_aliases)
            cmd_types |= CommandInterpreter::eCommandTypesAliases;
        if (m_options.m_show_user_defined)
            cmd_types |= CommandInterpreter::eCommandTypesUserDef;

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
        
        if (cmd_obj != NULL)
        {
            StringList matches;
            bool all_okay = true;
            CommandObject *sub_cmd_obj = cmd_obj;
            // Loop down through sub_command dictionaries until we find the command object that corresponds
            // to the help command entered.
            for (int i = 1; i < argc && all_okay; ++i)
            {
                std::string sub_command = command.GetArgumentAtIndex(i);
                matches.Clear();
                if (! sub_cmd_obj->IsMultiwordObject ())
                {
                    all_okay = false;
                }
                else
                {
                    CommandObject *found_cmd;
                    found_cmd = sub_cmd_obj->GetSubcommandObject(sub_command.c_str(), &matches);
                    if (found_cmd == NULL)
                        all_okay = false;
                    else if (matches.GetSize() > 1)
                        all_okay = false;
                    else
                        sub_cmd_obj = found_cmd;
                }
            }
            
            if (!all_okay || (sub_cmd_obj == NULL))
            {
                std::string cmd_string;
                command.GetCommandString (cmd_string);
                if (matches.GetSize() < 2)
                {
                    result.AppendErrorWithFormat("'%s' is not a known command.\n"
                                                 "Try 'help' to see a current list of commands.\n",
                                                 cmd_string.c_str());
                }
                else 
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
                }

                result.SetStatus (eReturnStatusFailed);
            }
            else
            {
                Stream &output_strm = result.GetOutputStream();
                if (sub_cmd_obj->GetOptions() != NULL)
                {
                    if (sub_cmd_obj->WantsRawCommandString())
                    {
                        std::string help_text (sub_cmd_obj->GetHelp());
                        help_text.append ("  This command takes 'raw' input (no need to quote stuff).");
                        m_interpreter.OutputFormattedHelpText (output_strm, "", "", help_text.c_str(), 1);
                    }
                    else
                        m_interpreter.OutputFormattedHelpText (output_strm, "", "", sub_cmd_obj->GetHelp(), 1);
                    output_strm.Printf ("\nSyntax: %s\n", sub_cmd_obj->GetSyntax());
                    sub_cmd_obj->GetOptions()->GenerateOptionUsage (output_strm, sub_cmd_obj);
                    const char *long_help = sub_cmd_obj->GetHelpLong();
                    if ((long_help != NULL)
                        && (strlen (long_help) > 0))
                        output_strm.Printf ("\n%s", long_help);
                    if (sub_cmd_obj->WantsRawCommandString() && !sub_cmd_obj->WantsCompletion())
                    {
                        // Emit the message about using ' -- ' between the end of the command options and the raw input
                        // conditionally, i.e., only if the command object does not want completion.
                        m_interpreter.OutputFormattedHelpText (output_strm, "", "",
                                                               "\nIMPORTANT NOTE:  Because this command takes 'raw' input, if you use any command options"
                                                               " you must use ' -- ' between the end of the command options and the beginning of the raw input.", 1);
                    }
                    else if (sub_cmd_obj->GetNumArgumentEntries() > 0
                             && sub_cmd_obj->GetOptions()
                             && sub_cmd_obj->GetOptions()->NumCommandOptions() > 0)
                    {
                            // Also emit a warning about using "--" in case you are using a command that takes options and arguments.
                            m_interpreter.OutputFormattedHelpText (output_strm, "", "",
                                                                   "\nThis command takes options and arguments, if your arguments look like option specifiers"
                                                                   " you must use '--' to terminate the options before starting to give the arguments.", 1);
                    }

                    // Mark this help command with a success status.
                    result.SetStatus (eReturnStatusSuccessFinishNoResult);
                }
                else if (sub_cmd_obj->IsMultiwordObject())
                {
                    if (sub_cmd_obj->WantsRawCommandString())
                    {
                        std::string help_text (sub_cmd_obj->GetHelp());
                        help_text.append ("  This command takes 'raw' input (no need to quote stuff).");
                        m_interpreter.OutputFormattedHelpText (output_strm, "", "", help_text.c_str(), 1);
                    }
                    else
                        m_interpreter.OutputFormattedHelpText (output_strm, "", "", sub_cmd_obj->GetHelp(), 1);
                    sub_cmd_obj->GenerateHelpText (result);
                }
                else
                {
                    const char *long_help = sub_cmd_obj->GetHelpLong();
                    if ((long_help != NULL)
                        && (strlen (long_help) > 0))
                        output_strm.Printf ("%s", long_help);
                    else if (sub_cmd_obj->WantsRawCommandString())
                    {
                        std::string help_text (sub_cmd_obj->GetHelp());
                        help_text.append ("  This command takes 'raw' input (no need to quote stuff).");
                        m_interpreter.OutputFormattedHelpText (output_strm, "", "", help_text.c_str(), 1);
                    }
                    else
                        m_interpreter.OutputFormattedHelpText (output_strm, "", "", sub_cmd_obj->GetHelp(), 1);
                    output_strm.Printf ("\nSyntax: %s\n", sub_cmd_obj->GetSyntax());
                    // Mark this help command with a success status.
                    result.SetStatus (eReturnStatusSuccessFinishNoResult);
                }
            }
            
            if (is_alias_command)
            {
                StreamString sstr;
                m_interpreter.GetAliasHelp (alias_name.c_str(), cmd_obj->GetCommandName(), sstr);
                result.GetOutputStream().Printf ("\n'%s' is an abbreviation for %s\n", alias_name.c_str(), sstr.GetData());
            }
        }
        else if (matches.GetSize() > 0)
        {
            Stream &output_strm = result.GetOutputStream();
            output_strm.Printf("Help requested with ambiguous command name, possible completions:\n");
            const uint32_t match_count = matches.GetSize();
            for (uint32_t i = 0; i < match_count; i++)
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
                result.AppendErrorWithFormat 
                    ("'%s' is not a known command.\nTry 'help' to see a current list of commands.\n",
                     command.GetArgumentAtIndex(0));
                result.SetStatus (eReturnStatusFailed);
            }
        }
    }
    
    return result.Succeeded();
}

int
CommandObjectHelp::HandleCompletion
(
    Args &input,
    int &cursor_index,
    int &cursor_char_position,
    int match_start_point,
    int max_return_elements,
    bool &word_complete,
    StringList &matches
)
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
