//===-- CommandObjectWatchpointCommand.cpp ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

// C Includes
// C++ Includes


#include "CommandObjectWatchpointCommand.h"
#include "CommandObjectWatchpoint.h"

#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Breakpoint/Watchpoint.h"
#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Core/State.h"

#include <vector>

using namespace lldb;
using namespace lldb_private;

//-------------------------------------------------------------------------
// CommandObjectWatchpointCommandAdd
//-------------------------------------------------------------------------


class CommandObjectWatchpointCommandAdd : public CommandObjectParsed
{
public:

    CommandObjectWatchpointCommandAdd (CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter,
                             "add",
                             "Add a set of commands to a watchpoint, to be executed whenever the watchpoint is hit.",
                             NULL),
        m_options (interpreter)
    {
        SetHelpLong (
"\nGeneral information about entering watchpoint commands \n\
------------------------------------------------------ \n\
 \n\
This command will cause you to be prompted to enter the command or set \n\
of commands you wish to be executed when the specified watchpoint is \n\
hit.  You will be told to enter your command(s), and will see a '> ' \n\
prompt. Because you can enter one or many commands to be executed when \n\
a watchpoint is hit, you will continue to be prompted after each \n\
new-line that you enter, until you enter the word 'DONE', which will \n\
cause the commands you have entered to be stored with the watchpoint \n\
and executed when the watchpoint is hit. \n\
 \n\
Syntax checking is not necessarily done when watchpoint commands are \n\
entered.  An improperly written watchpoint command will attempt to get \n\
executed when the watchpoint gets hit, and usually silently fail.  If \n\
your watchpoint command does not appear to be getting executed, go \n\
back and check your syntax. \n\
 \n\
 \n\
Special information about PYTHON watchpoint commands                            \n\
----------------------------------------------------                            \n\
                                                                                \n\
You may enter either one line of Python or multiple lines of Python             \n\
(including defining whole functions, if desired).  If you enter a               \n\
single line of Python, that will be passed to the Python interpreter            \n\
'as is' when the watchpoint gets hit.  If you enter function                    \n\
definitions, they will be passed to the Python interpreter as soon as           \n\
you finish entering the watchpoint command, and they can be called              \n\
later (don't forget to add calls to them, if you want them called when          \n\
the watchpoint is hit).  If you enter multiple lines of Python that             \n\
are not function definitions, they will be collected into a new,                \n\
automatically generated Python function, and a call to the newly                \n\
generated function will be attached to the watchpoint.                          \n\
                                                                                \n\
This auto-generated function is passed in two arguments:                        \n\
                                                                                \n\
    frame:  an SBFrame object representing the frame which hit the watchpoint.  \n\
            From the frame you can get back to the thread and process.          \n\
    wp:     the watchpoint that was hit.                                        \n\
                                                                                \n\
Important Note: Because loose Python code gets collected into functions,        \n\
if you want to access global variables in the 'loose' code, you need to         \n\
specify that they are global, using the 'global' keyword.  Be sure to           \n\
use correct Python syntax, including indentation, when entering Python          \n\
watchpoint commands.                                                            \n\
                                                                                \n\
As a third option, you can pass the name of an already existing Python function \n\
and that function will be attached to the watchpoint. It will get passed the    \n\
frame and wp_loc arguments mentioned above.                                     \n\
                                                                                \n\
Example Python one-line watchpoint command: \n\
 \n\
(lldb) watchpoint command add -s python 1 \n\
Enter your Python command(s). Type 'DONE' to end. \n\
> print \"Hit this watchpoint!\" \n\
> DONE \n\
 \n\
As a convenience, this also works for a short Python one-liner: \n\
(lldb) watchpoint command add -s python 1 -o \"import time; print time.asctime()\" \n\
(lldb) run \n\
Launching '.../a.out'  (x86_64) \n\
(lldb) Fri Sep 10 12:17:45 2010 \n\
Process 21778 Stopped \n\
* thread #1: tid = 0x2e03, 0x0000000100000de8 a.out`c + 7 at main.c:39, stop reason = watchpoint 1.1, queue = com.apple.main-thread \n\
  36   	\n\
  37   	int c(int val)\n\
  38   	{\n\
  39 ->	    return val + 3;\n\
  40   	}\n\
  41   	\n\
  42   	int main (int argc, char const *argv[])\n\
(lldb) \n\
 \n\
Example multiple line Python watchpoint command, using function definition: \n\
 \n\
(lldb) watchpoint command add -s python 1 \n\
Enter your Python command(s). Type 'DONE' to end. \n\
> def watchpoint_output (wp_no): \n\
>     out_string = \"Hit watchpoint number \" + repr (wp_no) \n\
>     print out_string \n\
>     return True \n\
> watchpoint_output (1) \n\
> DONE \n\
 \n\
 \n\
Example multiple line Python watchpoint command, using 'loose' Python: \n\
 \n\
(lldb) watchpoint command add -s p 1 \n\
Enter your Python command(s). Type 'DONE' to end. \n\
> global wp_count \n\
> wp_count = wp_count + 1 \n\
> print \"Hit this watchpoint \" + repr(wp_count) + \" times!\" \n\
> DONE \n\
 \n\
In this case, since there is a reference to a global variable, \n\
'wp_count', you will also need to make sure 'wp_count' exists and is \n\
initialized: \n\
 \n\
(lldb) script \n\
>>> wp_count = 0 \n\
>>> quit() \n\
 \n\
(lldb)  \n\
 \n\
 \n\
Final Note:  If you get a warning that no watchpoint command was generated, \n\
but you did not get any syntax errors, you probably forgot to add a call \n\
to your functions. \n\
 \n\
Special information about debugger command watchpoint commands \n\
-------------------------------------------------------------- \n\
 \n\
You may enter any debugger command, exactly as you would at the \n\
debugger prompt.  You may enter as many debugger commands as you like, \n\
but do NOT enter more than one command per line. \n" );

        CommandArgumentEntry arg;
        CommandArgumentData wp_id_arg;

        // Define the first (and only) variant of this arg.
        wp_id_arg.arg_type = eArgTypeWatchpointID;
        wp_id_arg.arg_repetition = eArgRepeatPlain;

        // There is only one variant this argument could be; put it into the argument entry.
        arg.push_back (wp_id_arg);

        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg);
    }

    virtual
    ~CommandObjectWatchpointCommandAdd () {}

    virtual Options *
    GetOptions ()
    {
        return &m_options;
    }

    void
    CollectDataForWatchpointCommandCallback (WatchpointOptions *wp_options, 
                                             CommandReturnObject &result)
    {
        InputReaderSP reader_sp (new InputReader(m_interpreter.GetDebugger()));
        std::unique_ptr<WatchpointOptions::CommandData> data_ap(new WatchpointOptions::CommandData());
        if (reader_sp && data_ap.get())
        {
            BatonSP baton_sp (new WatchpointOptions::CommandBaton (data_ap.release()));
            wp_options->SetCallback (WatchpointOptionsCallbackFunction, baton_sp);

            Error err (reader_sp->Initialize (CommandObjectWatchpointCommandAdd::GenerateWatchpointCommandCallback,
                                              wp_options,                   // callback_data
                                              eInputReaderGranularityLine,  // token size, to pass to callback function
                                              "DONE",                       // end token
                                              "> ",                         // prompt
                                              true));                       // echo input
            if (err.Success())
            {
                m_interpreter.GetDebugger().PushInputReader (reader_sp);
                result.SetStatus (eReturnStatusSuccessFinishNoResult);
            }
            else
            {
                result.AppendError (err.AsCString());
                result.SetStatus (eReturnStatusFailed);
            }
        }
        else
        {
            result.AppendError("out of memory");
            result.SetStatus (eReturnStatusFailed);
        }

    }
    
    /// Set a one-liner as the callback for the watchpoint.
    void 
    SetWatchpointCommandCallback (WatchpointOptions *wp_options,
                                  const char *oneliner)
    {
        std::unique_ptr<WatchpointOptions::CommandData> data_ap(new WatchpointOptions::CommandData());

        // It's necessary to set both user_source and script_source to the oneliner.
        // The former is used to generate callback description (as in watchpoint command list)
        // while the latter is used for Python to interpret during the actual callback.
        data_ap->user_source.AppendString (oneliner);
        data_ap->script_source.assign (oneliner);
        data_ap->stop_on_error = m_options.m_stop_on_error;

        BatonSP baton_sp (new WatchpointOptions::CommandBaton (data_ap.release()));
        wp_options->SetCallback (WatchpointOptionsCallbackFunction, baton_sp);

        return;
    }

    static size_t
    GenerateWatchpointCommandCallback (void *callback_data, 
                                       InputReader &reader, 
                                       lldb::InputReaderAction notification,
                                       const char *bytes, 
                                       size_t bytes_len)
    {
        StreamSP out_stream = reader.GetDebugger().GetAsyncOutputStream();
        bool batch_mode = reader.GetDebugger().GetCommandInterpreter().GetBatchCommandMode();
        
        switch (notification)
        {
        case eInputReaderActivate:
            if (!batch_mode)
            {
                out_stream->Printf ("%s\n", g_reader_instructions);
                if (reader.GetPrompt())
                    out_stream->Printf ("%s", reader.GetPrompt());
                out_stream->Flush();
            }
            break;

        case eInputReaderDeactivate:
            break;

        case eInputReaderReactivate:
            if (reader.GetPrompt() && !batch_mode)
            {
                out_stream->Printf ("%s", reader.GetPrompt());
                out_stream->Flush();
            }
            break;

        case eInputReaderAsynchronousOutputWritten:
            break;
            
        case eInputReaderGotToken:
            if (bytes && bytes_len && callback_data)
            {
                WatchpointOptions *wp_options = (WatchpointOptions *) callback_data;
                if (wp_options)
                {
                    Baton *wp_options_baton = wp_options->GetBaton();
                    if (wp_options_baton)
                        ((WatchpointOptions::CommandData *)wp_options_baton->m_data)->user_source.AppendString (bytes, bytes_len); 
                }
            }
            if (!reader.IsDone() && reader.GetPrompt() && !batch_mode)
            {
                out_stream->Printf ("%s", reader.GetPrompt());
                out_stream->Flush();
            }
            break;
            
        case eInputReaderInterrupt:
            {
                // Finish, and cancel the watchpoint command.
                reader.SetIsDone (true);
                WatchpointOptions *wp_options = (WatchpointOptions *) callback_data;
                if (wp_options)
                {
                    Baton *wp_options_baton = wp_options->GetBaton ();
                    if (wp_options_baton)
                    {
                        ((WatchpointOptions::CommandData *) wp_options_baton->m_data)->user_source.Clear();
                        ((WatchpointOptions::CommandData *) wp_options_baton->m_data)->script_source.clear();
                    }
                }
                if (!batch_mode)
                {
                    out_stream->Printf ("Warning: No command attached to watchpoint.\n");
                    out_stream->Flush();
                }
            }
            break;
            
        case eInputReaderEndOfFile:
            reader.SetIsDone (true);
            break;
            
        case eInputReaderDone:
            break;
        }

        return bytes_len;
    }
    
    static bool
    WatchpointOptionsCallbackFunction (void *baton,
                                       StoppointCallbackContext *context, 
                                       lldb::user_id_t watch_id)
    {
        bool ret_value = true;
        if (baton == NULL)
            return true;
        
        
        WatchpointOptions::CommandData *data = (WatchpointOptions::CommandData *) baton;
        StringList &commands = data->user_source;
        
        if (commands.GetSize() > 0)
        {
            ExecutionContext exe_ctx (context->exe_ctx_ref);
            Target *target = exe_ctx.GetTargetPtr();
            if (target)
            {
                CommandReturnObject result;
                Debugger &debugger = target->GetDebugger();
                // Rig up the results secondary output stream to the debugger's, so the output will come out synchronously
                // if the debugger is set up that way.
                    
                StreamSP output_stream (debugger.GetAsyncOutputStream());
                StreamSP error_stream (debugger.GetAsyncErrorStream());
                result.SetImmediateOutputStream (output_stream);
                result.SetImmediateErrorStream (error_stream);
        
                bool stop_on_continue = true;
                bool echo_commands    = false;
                bool print_results    = true;

                debugger.GetCommandInterpreter().HandleCommands (commands, 
                                                                 &exe_ctx,
                                                                 stop_on_continue, 
                                                                 data->stop_on_error, 
                                                                 echo_commands, 
                                                                 print_results,
                                                                 eLazyBoolNo,
                                                                 result);
                result.GetImmediateOutputStream()->Flush();
                result.GetImmediateErrorStream()->Flush();
           }
        }
        return ret_value;
    }    

    class CommandOptions : public Options
    {
    public:

        CommandOptions (CommandInterpreter &interpreter) :
            Options (interpreter),
            m_use_commands (false),
            m_use_script_language (false),
            m_script_language (eScriptLanguageNone),
            m_use_one_liner (false),
            m_one_liner(),
            m_function_name()
        {
        }

        virtual
        ~CommandOptions () {}

        virtual Error
        SetOptionValue (uint32_t option_idx, const char *option_arg)
        {
            Error error;
            const int short_option = m_getopt_table[option_idx].val;

            switch (short_option)
            {
            case 'o':
                m_use_one_liner = true;
                m_one_liner = option_arg;
                break;

            case 's':
                m_script_language = (lldb::ScriptLanguage) Args::StringToOptionEnum (option_arg, 
                                                                                     g_option_table[option_idx].enum_values, 
                                                                                     eScriptLanguageNone,
                                                                                     error);

                if (m_script_language == eScriptLanguagePython || m_script_language == eScriptLanguageDefault)
                {
                    m_use_script_language = true;
                }
                else
                {
                    m_use_script_language = false;
                }          
                break;

            case 'e':
                {
                    bool success = false;
                    m_stop_on_error = Args::StringToBoolean(option_arg, false, &success);
                    if (!success)
                        error.SetErrorStringWithFormat("invalid value for stop-on-error: \"%s\"", option_arg);
                }
                break;
                    
            case 'F':
                {
                    m_use_one_liner = false;
                    m_use_script_language = true;
                    m_function_name.assign(option_arg);
                }
                break;

            default:
                break;
            }
            return error;
        }
        void
        OptionParsingStarting ()
        {
            m_use_commands = true;
            m_use_script_language = false;
            m_script_language = eScriptLanguageNone;

            m_use_one_liner = false;
            m_stop_on_error = true;
            m_one_liner.clear();
            m_function_name.clear();
        }

        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }

        // Options table: Required for subclasses of Options.

        static OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.

        bool m_use_commands;
        bool m_use_script_language;
        lldb::ScriptLanguage m_script_language;

        // Instance variables to hold the values for one_liner options.
        bool m_use_one_liner;
        std::string m_one_liner;
        bool m_stop_on_error;
        std::string m_function_name;
    };

protected:
    virtual bool
    DoExecute (Args& command, CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();

        if (target == NULL)
        {
            result.AppendError ("There is not a current executable; there are no watchpoints to which to add commands");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        const WatchpointList &watchpoints = target->GetWatchpointList();
        size_t num_watchpoints = watchpoints.GetSize();

        if (num_watchpoints == 0)
        {
            result.AppendError ("No watchpoints exist to have commands added");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        if (m_options.m_use_script_language == false && m_options.m_function_name.size())
        {
            result.AppendError ("need to enable scripting to have a function run as a watchpoint command");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        
        std::vector<uint32_t> valid_wp_ids;
        if (!CommandObjectMultiwordWatchpoint::VerifyWatchpointIDs(command, valid_wp_ids))
        {
            result.AppendError("Invalid watchpoints specification.");
            result.SetStatus(eReturnStatusFailed);
            return false;
        }

        result.SetStatus(eReturnStatusSuccessFinishNoResult);
        const size_t count = valid_wp_ids.size();
        for (size_t i = 0; i < count; ++i)
        {
            uint32_t cur_wp_id = valid_wp_ids.at (i);
            if (cur_wp_id != LLDB_INVALID_WATCH_ID)
            {
                Watchpoint *wp = target->GetWatchpointList().FindByID (cur_wp_id).get();
                // Sanity check wp first.
                if (wp == NULL) continue;

                WatchpointOptions *wp_options = wp->GetOptions();
                // Skip this watchpoint if wp_options is not good.
                if (wp_options == NULL) continue;

                // If we are using script language, get the script interpreter
                // in order to set or collect command callback.  Otherwise, call
                // the methods associated with this object.
                if (m_options.m_use_script_language)
                {
                    // Special handling for one-liner specified inline.
                    if (m_options.m_use_one_liner)
                    {
                        m_interpreter.GetScriptInterpreter()->SetWatchpointCommandCallback (wp_options,
                                                                                            m_options.m_one_liner.c_str());
                    }
                    // Special handling for using a Python function by name
                    // instead of extending the watchpoint callback data structures, we just automatize
                    // what the user would do manually: make their watchpoint command be a function call
                    else if (m_options.m_function_name.size())
                    {
                        std::string oneliner(m_options.m_function_name);
                        oneliner += "(frame, wp, internal_dict)";
                        m_interpreter.GetScriptInterpreter()->SetWatchpointCommandCallback (wp_options,
                                                                                            oneliner.c_str());
                    }
                    else
                    {
                        m_interpreter.GetScriptInterpreter()->CollectDataForWatchpointCommandCallback (wp_options,
                                                                                                       result);
                    }
                }
                else
                {
                    // Special handling for one-liner specified inline.
                    if (m_options.m_use_one_liner)
                        SetWatchpointCommandCallback (wp_options,
                                                      m_options.m_one_liner.c_str());
                    else
                        CollectDataForWatchpointCommandCallback (wp_options, 
                                                                 result);
                }
            }
        }

        return result.Succeeded();
    }

private:
    CommandOptions m_options;
    static const char *g_reader_instructions;

};

const char *
CommandObjectWatchpointCommandAdd::g_reader_instructions = "Enter your debugger command(s).  Type 'DONE' to end.";

// FIXME: "script-type" needs to have its contents determined dynamically, so somebody can add a new scripting
// language to lldb and have it pickable here without having to change this enumeration by hand and rebuild lldb proper.

static OptionEnumValueElement
g_script_option_enumeration[4] =
{
    { eScriptLanguageNone,    "command",         "Commands are in the lldb command interpreter language"},
    { eScriptLanguagePython,  "python",          "Commands are in the Python language."},
    { eSortOrderByName,       "default-script",  "Commands are in the default scripting language."},
    { 0,                      NULL,              NULL }
};

OptionDefinition
CommandObjectWatchpointCommandAdd::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_1,   false, "one-liner",       'o', required_argument, NULL, 0, eArgTypeOneLiner,
        "Specify a one-line watchpoint command inline. Be sure to surround it with quotes." },

    { LLDB_OPT_SET_ALL, false, "stop-on-error",   'e', required_argument, NULL, 0, eArgTypeBoolean,
        "Specify whether watchpoint command execution should terminate on error." },

    { LLDB_OPT_SET_ALL, false, "script-type",     's', required_argument, g_script_option_enumeration, 0, eArgTypeNone,
        "Specify the language for the commands - if none is specified, the lldb command interpreter will be used."},

    { LLDB_OPT_SET_2,   false, "python-function", 'F', required_argument, NULL, 0, eArgTypePythonFunction,
        "Give the name of a Python function to run as command for this watchpoint. Be sure to give a module name if appropriate."},
    
    { 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};

//-------------------------------------------------------------------------
// CommandObjectWatchpointCommandDelete
//-------------------------------------------------------------------------

class CommandObjectWatchpointCommandDelete : public CommandObjectParsed
{
public:
    CommandObjectWatchpointCommandDelete (CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter, 
                             "delete",
                             "Delete the set of commands from a watchpoint.",
                             NULL)
    {
        CommandArgumentEntry arg;
        CommandArgumentData wp_id_arg;

        // Define the first (and only) variant of this arg.
        wp_id_arg.arg_type = eArgTypeWatchpointID;
        wp_id_arg.arg_repetition = eArgRepeatPlain;

        // There is only one variant this argument could be; put it into the argument entry.
        arg.push_back (wp_id_arg);

        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg);
    }


    virtual
    ~CommandObjectWatchpointCommandDelete () {}

protected:
    virtual bool
    DoExecute (Args& command, CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();

        if (target == NULL)
        {
            result.AppendError ("There is not a current executable; there are no watchpoints from which to delete commands");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        const WatchpointList &watchpoints = target->GetWatchpointList();
        size_t num_watchpoints = watchpoints.GetSize();

        if (num_watchpoints == 0)
        {
            result.AppendError ("No watchpoints exist to have commands deleted");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        if (command.GetArgumentCount() == 0)
        {
            result.AppendError ("No watchpoint specified from which to delete the commands");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        std::vector<uint32_t> valid_wp_ids;
        if (!CommandObjectMultiwordWatchpoint::VerifyWatchpointIDs(command, valid_wp_ids))
        {
            result.AppendError("Invalid watchpoints specification.");
            result.SetStatus(eReturnStatusFailed);
            return false;
        }

        result.SetStatus(eReturnStatusSuccessFinishNoResult);
        const size_t count = valid_wp_ids.size();
        for (size_t i = 0; i < count; ++i)
        {
            uint32_t cur_wp_id = valid_wp_ids.at (i);
            if (cur_wp_id != LLDB_INVALID_WATCH_ID)
            {
                Watchpoint *wp = target->GetWatchpointList().FindByID (cur_wp_id).get();
                if (wp)
                    wp->ClearCallback();
            }
            else
            {
                result.AppendErrorWithFormat("Invalid watchpoint ID: %u.\n", 
                                             cur_wp_id);
                result.SetStatus (eReturnStatusFailed);
                return false;
            }
        }
        return result.Succeeded();
    }
};

//-------------------------------------------------------------------------
// CommandObjectWatchpointCommandList
//-------------------------------------------------------------------------

class CommandObjectWatchpointCommandList : public CommandObjectParsed
{
public:
    CommandObjectWatchpointCommandList (CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter,
                             "list",
                             "List the script or set of commands to be executed when the watchpoint is hit.",
                              NULL)
    {
        CommandArgumentEntry arg;
        CommandArgumentData wp_id_arg;

        // Define the first (and only) variant of this arg.
        wp_id_arg.arg_type = eArgTypeWatchpointID;
        wp_id_arg.arg_repetition = eArgRepeatPlain;

        // There is only one variant this argument could be; put it into the argument entry.
        arg.push_back (wp_id_arg);

        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg);
    }

    virtual
    ~CommandObjectWatchpointCommandList () {}

protected:
    virtual bool
    DoExecute (Args& command,
             CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();

        if (target == NULL)
        {
            result.AppendError ("There is not a current executable; there are no watchpoints for which to list commands");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        const WatchpointList &watchpoints = target->GetWatchpointList();
        size_t num_watchpoints = watchpoints.GetSize();

        if (num_watchpoints == 0)
        {
            result.AppendError ("No watchpoints exist for which to list commands");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        if (command.GetArgumentCount() == 0)
        {
            result.AppendError ("No watchpoint specified for which to list the commands");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        std::vector<uint32_t> valid_wp_ids;
        if (!CommandObjectMultiwordWatchpoint::VerifyWatchpointIDs(command, valid_wp_ids))
        {
            result.AppendError("Invalid watchpoints specification.");
            result.SetStatus(eReturnStatusFailed);
            return false;
        }

        result.SetStatus(eReturnStatusSuccessFinishNoResult);
        const size_t count = valid_wp_ids.size();
        for (size_t i = 0; i < count; ++i)
        {
            uint32_t cur_wp_id = valid_wp_ids.at (i);
            if (cur_wp_id != LLDB_INVALID_WATCH_ID)
            {
                Watchpoint *wp = target->GetWatchpointList().FindByID (cur_wp_id).get();
                
                if (wp)
                {
                    const WatchpointOptions *wp_options = wp->GetOptions();
                    if (wp_options)
                    {
                        // Get the callback baton associated with the current watchpoint.
                        const Baton *baton = wp_options->GetBaton();
                        if (baton)
                        {
                            result.GetOutputStream().Printf ("Watchpoint %u:\n", cur_wp_id);
                            result.GetOutputStream().IndentMore ();
                            baton->GetDescription(&result.GetOutputStream(), eDescriptionLevelFull);
                            result.GetOutputStream().IndentLess ();
                        }
                        else
                        {
                            result.AppendMessageWithFormat ("Watchpoint %u does not have an associated command.\n", 
                                                            cur_wp_id);
                        }
                    }
                    result.SetStatus (eReturnStatusSuccessFinishResult);
                }
                else
                {
                    result.AppendErrorWithFormat("Invalid watchpoint ID: %u.\n", cur_wp_id);
                    result.SetStatus (eReturnStatusFailed);
                }
            }
        }

        return result.Succeeded();
    }
};

//-------------------------------------------------------------------------
// CommandObjectWatchpointCommand
//-------------------------------------------------------------------------

CommandObjectWatchpointCommand::CommandObjectWatchpointCommand (CommandInterpreter &interpreter) :
    CommandObjectMultiword (interpreter,
                            "command",
                            "A set of commands for adding, removing and examining bits of code to be executed when the watchpoint is hit (watchpoint 'commmands').",
                            "command <sub-command> [<sub-command-options>] <watchpoint-id>")
{
    CommandObjectSP add_command_object (new CommandObjectWatchpointCommandAdd (interpreter));
    CommandObjectSP delete_command_object (new CommandObjectWatchpointCommandDelete (interpreter));
    CommandObjectSP list_command_object (new CommandObjectWatchpointCommandList (interpreter));

    add_command_object->SetCommandName ("watchpoint command add");
    delete_command_object->SetCommandName ("watchpoint command delete");
    list_command_object->SetCommandName ("watchpoint command list");

    LoadSubCommand ("add",    add_command_object);
    LoadSubCommand ("delete", delete_command_object);
    LoadSubCommand ("list",   list_command_object);
}

CommandObjectWatchpointCommand::~CommandObjectWatchpointCommand ()
{
}


