//===-- CommandObjectPlugin.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include "CommandObjectPlugin.h"

#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBCommandInterpreter.h"
#include "lldb/API/SBCommandReturnObject.h"

#include "lldb/Host/Host.h"

#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"

using namespace lldb;
using namespace lldb_private;

class CommandObjectPluginLoad : public CommandObjectParsed
{
private:
public:
    CommandObjectPluginLoad (CommandInterpreter &interpreter) :
    CommandObjectParsed (interpreter,
                         "plugin load",
                         "Import a dylib that implements an LLDB plugin.",
                         NULL)
    {
        CommandArgumentEntry arg1;
        CommandArgumentData cmd_arg;
        
        // Define the first (and only) variant of this arg.
        cmd_arg.arg_type = eArgTypeFilename;
        cmd_arg.arg_repetition = eArgRepeatPlain;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg1.push_back (cmd_arg);
        
        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg1);
    }
    
    ~CommandObjectPluginLoad ()
    {
    }
    
    int
    HandleArgumentCompletion (Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches)
    {
        std::string completion_str (input.GetArgumentAtIndex(cursor_index));
        completion_str.erase (cursor_char_position);
        
        CommandCompletions::InvokeCommonCompletionCallbacks (m_interpreter,
                                                             CommandCompletions::eDiskFileCompletion,
                                                             completion_str.c_str(),
                                                             match_start_point,
                                                             max_return_elements,
                                                             NULL,
                                                             word_complete,
                                                             matches);
        return matches.GetSize();
    }

protected:
    bool
    DoExecute (Args& command, CommandReturnObject &result)
    {
        typedef void (*LLDBCommandPluginInit) (lldb::SBDebugger debugger);
        
        size_t argc = command.GetArgumentCount();
        
        if (argc != 1)
        {
            result.AppendError ("'plugin load' requires one argument");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        
        const char* path = command.GetArgumentAtIndex(0);
        
        Error error;
        
        FileSpec dylib_fspec(path,true);
        
        if (m_interpreter.GetDebugger().LoadPlugin(dylib_fspec, error))
            result.SetStatus(eReturnStatusSuccessFinishResult);
        else
        {
            result.AppendError(error.AsCString());
            result.SetStatus(eReturnStatusFailed);
        }
        
        return result.Succeeded();
    }
};

CommandObjectPlugin::CommandObjectPlugin (CommandInterpreter &interpreter) :
CommandObjectMultiword (interpreter,
                        "plugin",
                        "A set of commands for managing or customizing plugin commands.",
                        "plugin <subcommand> [<subcommand-options>]")
{
    LoadSubCommand ("load",  CommandObjectSP (new CommandObjectPluginLoad (interpreter)));
}
    
CommandObjectPlugin::~CommandObjectPlugin ()
{
}
