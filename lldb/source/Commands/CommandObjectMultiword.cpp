//===-- CommandObjectMultiword.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/CommandObjectMultiword.h"
// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Debugger.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Interpreter/CommandReturnObject.h"

using namespace lldb;
using namespace lldb_private;

//-------------------------------------------------------------------------
// CommandObjectMultiword
//-------------------------------------------------------------------------

CommandObjectMultiword::CommandObjectMultiword
(
    CommandInterpreter &interpreter,
    const char *name,
    const char *help,
    const char *syntax,
    uint32_t flags
) :
    CommandObject (interpreter, name, help, syntax, flags)
{
}

CommandObjectMultiword::~CommandObjectMultiword ()
{
}

CommandObjectSP
CommandObjectMultiword::GetSubcommandSP (const char *sub_cmd, StringList *matches)
{
    CommandObjectSP return_cmd_sp;
    CommandObject::CommandMap::iterator pos;

    if (!m_subcommand_dict.empty())
    {
        pos = m_subcommand_dict.find (sub_cmd);
        if (pos != m_subcommand_dict.end()) {
            // An exact match; append the sub_cmd to the 'matches' string list.
            if (matches)
                matches->AppendString(sub_cmd);
            return_cmd_sp = pos->second;
        }
        else
        {

            StringList local_matches;
            if (matches == NULL)
                matches = &local_matches;
            int num_matches = CommandObject::AddNamesMatchingPartialString (m_subcommand_dict, sub_cmd, *matches);

            if (num_matches == 1)
            {
                // Cleaner, but slightly less efficient would be to call back into this function, since I now
                // know I have an exact match...

                sub_cmd = matches->GetStringAtIndex(0);
                pos = m_subcommand_dict.find(sub_cmd);
                if (pos != m_subcommand_dict.end())
                    return_cmd_sp = pos->second;
            }
        }
    }
    return return_cmd_sp;
}

CommandObject *
CommandObjectMultiword::GetSubcommandObject (const char *sub_cmd, StringList *matches)
{
    return GetSubcommandSP(sub_cmd, matches).get();
}

bool
CommandObjectMultiword::LoadSubCommand 
(
    const char *name,
    const CommandObjectSP& cmd_obj
)
{
    CommandMap::iterator pos;
    bool success = true;

    pos = m_subcommand_dict.find(name);
    if (pos == m_subcommand_dict.end())
    {
        m_subcommand_dict[name] = cmd_obj;
        m_interpreter.CrossRegisterCommand (name, GetCommandName());
    }
    else
        success = false;

    return success;
}

bool
CommandObjectMultiword::Execute(const char *args_string, CommandReturnObject &result)
{
    Args args (args_string);
    const size_t argc = args.GetArgumentCount();
    if (argc == 0)
    {
        GenerateHelpText (result);
    }
    else
    {
        const char *sub_command = args.GetArgumentAtIndex (0);

        if (sub_command)
        {
            if (::strcasecmp (sub_command, "help") == 0)
            {
                GenerateHelpText (result);
            }
            else if (!m_subcommand_dict.empty())
            {
                StringList matches;
                CommandObject *sub_cmd_obj = GetSubcommandObject(sub_command, &matches);
                if (sub_cmd_obj != NULL)
                {
                    // Now call CommandObject::Execute to process and options in 'rest_of_line'.  From there
                    // the command-specific version of Execute will be called, with the processed arguments.

                    args.Shift();

                    sub_cmd_obj->Execute (args_string, result);
                }
                else
                {
                    std::string error_msg;
                    int num_subcmd_matches = matches.GetSize();
                    if (num_subcmd_matches > 0)
                        error_msg.assign ("ambiguous command ");
                    else
                        error_msg.assign ("invalid command ");

                    error_msg.append ("'");
                    error_msg.append (GetCommandName());
                    error_msg.append (" ");
                    error_msg.append (sub_command);
                    error_msg.append ("'");

                    if (num_subcmd_matches > 0)
                    {
                        error_msg.append (" Possible completions:");
                        for (int i = 0; i < num_subcmd_matches; i++)
                        {
                            error_msg.append ("\n\t");
                            error_msg.append (matches.GetStringAtIndex (i));
                        }
                    }
                    error_msg.append ("\n");
                    result.AppendRawError (error_msg.c_str(), error_msg.size());
                    result.SetStatus (eReturnStatusFailed);
                }
            }
            else
            {
                result.AppendErrorWithFormat ("'%s' does not have any subcommands.\n", GetCommandName());
                result.SetStatus (eReturnStatusFailed);
            }
        }
    }

    return result.Succeeded();
}

void
CommandObjectMultiword::GenerateHelpText (CommandReturnObject &result)
{
    // First time through here, generate the help text for the object and
    // push it to the return result object as well

    Stream &output_stream = result.GetOutputStream();
    output_stream.PutCString ("The following subcommands are supported:\n\n");

    CommandMap::iterator pos;
    uint32_t max_len = m_interpreter.FindLongestCommandWord (m_subcommand_dict);

    if (max_len)
        max_len += 4; // Indent the output by 4 spaces.

    for (pos = m_subcommand_dict.begin(); pos != m_subcommand_dict.end(); ++pos)
    {
        std::string indented_command ("    ");
        indented_command.append (pos->first);
        if (pos->second->WantsRawCommandString ())
        {
            std::string help_text (pos->second->GetHelp());
            help_text.append ("  This command takes 'raw' input (no need to quote stuff).");
            m_interpreter.OutputFormattedHelpText (result.GetOutputStream(),
                                                   indented_command.c_str(),
                                                   "--",
                                                   help_text.c_str(),
                                                   max_len);
        }
        else
            m_interpreter.OutputFormattedHelpText (result.GetOutputStream(), 
                                                   indented_command.c_str(),
                                                   "--", 
                                                   pos->second->GetHelp(), 
                                                   max_len);
    }

    output_stream.PutCString ("\nFor more help on any particular subcommand, type 'help <command> <subcommand>'.\n");

    result.SetStatus (eReturnStatusSuccessFinishNoResult);
}

int
CommandObjectMultiword::HandleCompletion
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
    // Any of the command matches will provide a complete word, otherwise the individual
    // completers will override this.
    word_complete = true;
    
    if (cursor_index == 0)
    {
        CommandObject::AddNamesMatchingPartialString (m_subcommand_dict, 
                                                      input.GetArgumentAtIndex(0), 
                                                      matches);

        if (matches.GetSize() == 1
            && matches.GetStringAtIndex(0) != NULL
            && strcmp (input.GetArgumentAtIndex(0), matches.GetStringAtIndex(0)) == 0)
        {
            StringList temp_matches;
            CommandObject *cmd_obj = GetSubcommandObject (input.GetArgumentAtIndex(0), 
                                                          &temp_matches);
            if (cmd_obj != NULL)
            {
                matches.DeleteStringAtIndex (0);
                input.Shift();
                cursor_char_position = 0;
                input.AppendArgument ("");
                return cmd_obj->HandleCompletion (input, 
                                                  cursor_index, 
                                                  cursor_char_position, 
                                                  match_start_point,
                                                  max_return_elements,
                                                  word_complete, 
                                                  matches);
            }
            else
                return matches.GetSize();
        }
        else
            return matches.GetSize();
    }
    else
    {
        CommandObject *sub_command_object = GetSubcommandObject (input.GetArgumentAtIndex(0), 
                                                                 &matches);
        if (sub_command_object == NULL)
        {
            return matches.GetSize();
        }
        else
        {
            // Remove the one match that we got from calling GetSubcommandObject.
            matches.DeleteStringAtIndex(0);
            input.Shift();
            cursor_index--;
            return sub_command_object->HandleCompletion (input, 
                                                         cursor_index, 
                                                         cursor_char_position, 
                                                         match_start_point,
                                                         max_return_elements,
                                                         word_complete,
                                                         matches);
        }

    }
}

const char *
CommandObjectMultiword::GetRepeatCommand (Args &current_command_args, uint32_t index)
{
    index++;
    if (current_command_args.GetArgumentCount() <= index)
        return NULL;
    CommandObject *sub_command_object = GetSubcommandObject (current_command_args.GetArgumentAtIndex(index));
    if (sub_command_object == NULL)
        return NULL;
    return sub_command_object->GetRepeatCommand(current_command_args, index);
}


void
CommandObjectMultiword::AproposAllSubCommands (const char *prefix,
                                               const char *search_word,
                                               StringList &commands_found,
                                               StringList &commands_help)
{
    CommandObject::CommandMap::const_iterator pos;

    for (pos = m_subcommand_dict.begin(); pos != m_subcommand_dict.end(); ++pos)
    {
        const char * command_name = pos->first.c_str();
        CommandObject *sub_cmd_obj = pos->second.get();
        StreamString complete_command_name;
        
        complete_command_name.Printf ("%s %s", prefix, command_name);
        
        if (sub_cmd_obj->HelpTextContainsWord (search_word))
        {
            commands_found.AppendString (complete_command_name.GetData());
            commands_help.AppendString (sub_cmd_obj->GetHelp());
        }
        
        if (sub_cmd_obj->IsMultiwordObject())
            sub_cmd_obj->AproposAllSubCommands (complete_command_name.GetData(),
                                                search_word,
                                                commands_found,
                                                commands_help);
    }
}



CommandObjectProxy::CommandObjectProxy (CommandInterpreter &interpreter,
                                        const char *name,
                                        const char *help,
                                        const char *syntax,
                                        uint32_t flags) :
    CommandObject (interpreter, name, help, syntax, flags)
{
}

CommandObjectProxy::~CommandObjectProxy ()
{
}

const char *
CommandObjectProxy::GetHelpLong ()
{
    CommandObject *proxy_command = GetProxyCommandObject();
    if (proxy_command)
        return proxy_command->GetHelpLong();
    return NULL;
}

void
CommandObjectProxy::AddObject (const char *obj_name)
{
    CommandObject *proxy_command = GetProxyCommandObject();
    if (proxy_command)
        return proxy_command->AddObject (obj_name);
}

bool
CommandObjectProxy::IsCrossRefObject ()
{
    CommandObject *proxy_command = GetProxyCommandObject();
    if (proxy_command)
        return proxy_command->IsCrossRefObject();
    return false;
}

bool
CommandObjectProxy::IsRemovable() const
{
    const CommandObject *proxy_command = const_cast<CommandObjectProxy *>(this)->GetProxyCommandObject();
    if (proxy_command)
        return proxy_command->IsRemovable();
    return false;
}

bool
CommandObjectProxy::IsMultiwordObject ()
{
    CommandObject *proxy_command = GetProxyCommandObject();
    if (proxy_command)
        return proxy_command->IsMultiwordObject();
    return false;
}

lldb::CommandObjectSP
CommandObjectProxy::GetSubcommandSP (const char *sub_cmd, StringList *matches)
{
    CommandObject *proxy_command = GetProxyCommandObject();
    if (proxy_command)
        return proxy_command->GetSubcommandSP(sub_cmd, matches);
    return lldb::CommandObjectSP();
}

CommandObject *
CommandObjectProxy::GetSubcommandObject (const char *sub_cmd, StringList *matches)
{
    CommandObject *proxy_command = GetProxyCommandObject();
    if (proxy_command)
        return proxy_command->GetSubcommandObject(sub_cmd, matches);
    return NULL;
}

void
CommandObjectProxy::AproposAllSubCommands (const char *prefix,
                                           const char *search_word,
                                           StringList &commands_found,
                                           StringList &commands_help)
{
    CommandObject *proxy_command = GetProxyCommandObject();
    if (proxy_command)
        return proxy_command->AproposAllSubCommands (prefix,
                                                     search_word,
                                                     commands_found,
                                                     commands_help);
}

bool
CommandObjectProxy::LoadSubCommand (const char *cmd_name,
                                    const lldb::CommandObjectSP& command_sp)
{
    CommandObject *proxy_command = GetProxyCommandObject();
    if (proxy_command)
        return proxy_command->LoadSubCommand (cmd_name, command_sp);
    return false;
}

bool
CommandObjectProxy::WantsRawCommandString()
{
    CommandObject *proxy_command = GetProxyCommandObject();
    if (proxy_command)
        return proxy_command->WantsRawCommandString();
    return false;
}

bool
CommandObjectProxy::WantsCompletion()
{
    CommandObject *proxy_command = GetProxyCommandObject();
    if (proxy_command)
        return proxy_command->WantsCompletion();
    return false;
}


Options *
CommandObjectProxy::GetOptions ()
{
    CommandObject *proxy_command = GetProxyCommandObject();
    if (proxy_command)
        return proxy_command->GetOptions ();
    return NULL;
}


int
CommandObjectProxy::HandleCompletion (Args &input,
                                      int &cursor_index,
                                      int &cursor_char_position,
                                      int match_start_point,
                                      int max_return_elements,
                                      bool &word_complete,
                                      StringList &matches)
{
    CommandObject *proxy_command = GetProxyCommandObject();
    if (proxy_command)
        return proxy_command->HandleCompletion (input,
                                                cursor_index,
                                                cursor_char_position,
                                                match_start_point,
                                                max_return_elements,
                                                word_complete,
                                                matches);
    matches.Clear();
    return 0;
}
int
CommandObjectProxy::HandleArgumentCompletion (Args &input,
                                              int &cursor_index,
                                              int &cursor_char_position,
                                              OptionElementVector &opt_element_vector,
                                              int match_start_point,
                                              int max_return_elements,
                                              bool &word_complete,
                                              StringList &matches)
{
    CommandObject *proxy_command = GetProxyCommandObject();
    if (proxy_command)
        return proxy_command->HandleArgumentCompletion (input,
                                                        cursor_index,
                                                        cursor_char_position,
                                                        opt_element_vector,
                                                        match_start_point,
                                                        max_return_elements,
                                                        word_complete,
                                                        matches);
    matches.Clear();
    return 0;
}

const char *
CommandObjectProxy::GetRepeatCommand (Args &current_command_args,
                                      uint32_t index)
{
    CommandObject *proxy_command = GetProxyCommandObject();
    if (proxy_command)
        return proxy_command->GetRepeatCommand (current_command_args, index);
    return NULL;
}

bool
CommandObjectProxy::Execute (const char *args_string,
                             CommandReturnObject &result)
{
    CommandObject *proxy_command = GetProxyCommandObject();
    if (proxy_command)
        return proxy_command->Execute (args_string, result);
    result.AppendError ("command is not implemented");
    result.SetStatus (eReturnStatusFailed);
    return false;
}


