//===-- CommandObjectSyntax.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectSyntax.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/Args.h"
#include "lldb/Interpreter/Options.h"

#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/CommandObjectMultiword.h"

using namespace lldb;
using namespace lldb_private;

//-------------------------------------------------------------------------
// CommandObjectSyntax
//-------------------------------------------------------------------------

CommandObjectSyntax::CommandObjectSyntax (CommandInterpreter &interpreter) :
    CommandObjectParsed (interpreter,
                         "syntax",
                         "Shows the correct syntax for a given debugger command.",
                         "syntax <command>")
{
    CommandArgumentEntry arg;
    CommandArgumentData command_arg;

    // Define the first (and only) variant of this arg.
    command_arg.arg_type = eArgTypeCommandName;
    command_arg.arg_repetition = eArgRepeatPlain;

    // There is only one variant this argument could be; put it into the argument entry.
    arg.push_back (command_arg);

    // Push the data for the first argument into the m_arguments vector.
    m_arguments.push_back (arg);
}

CommandObjectSyntax::~CommandObjectSyntax()
{
}


bool
CommandObjectSyntax::DoExecute (Args& command, CommandReturnObject &result)
{
    CommandObject::CommandMap::iterator pos;
    CommandObject *cmd_obj;
    const int argc = command.GetArgumentCount();

    if (argc > 0)
    {
        cmd_obj = m_interpreter.GetCommandObject (command.GetArgumentAtIndex(0));
        bool all_okay = true;
        for (int i = 1; i < argc; ++i)
        {
            std::string sub_command = command.GetArgumentAtIndex (i);
            if (!cmd_obj->IsMultiwordObject())
                all_okay = false;
            else
            {
                cmd_obj = cmd_obj->GetSubcommandObject(sub_command.c_str());
                if (!cmd_obj)
                    all_okay = false;
            }
        }
        
        if (all_okay && (cmd_obj != NULL))
        {
            Stream &output_strm = result.GetOutputStream();
            if (cmd_obj->GetOptions() != NULL)
            {
                output_strm.Printf ("\nSyntax: %s\n", cmd_obj->GetSyntax());
                output_strm.Printf ("(Try 'help %s' for more information on command options syntax.)\n",
                                    cmd_obj->GetCommandName());
                result.SetStatus (eReturnStatusSuccessFinishNoResult);
            }
            else
            {
                output_strm.Printf ("\nSyntax: %s\n", cmd_obj->GetSyntax());
                result.SetStatus (eReturnStatusSuccessFinishNoResult);
            }
        }
        else
        {
            std::string cmd_string;
            command.GetCommandString (cmd_string);
            result.AppendErrorWithFormat ("'%s' is not a known command.\n", cmd_string.c_str());
            result.AppendError ("Try 'help' to see a current list of commands.");
            result.SetStatus (eReturnStatusFailed);
        }
    }
    else
    {
        result.AppendError ("Must call 'syntax' with a valid command.");
        result.SetStatus (eReturnStatusFailed);
    }

    return result.Succeeded();
}
