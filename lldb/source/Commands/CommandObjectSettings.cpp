//===-- CommandObjectSettings.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include "CommandObjectSettings.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/CommandCompletions.h"

using namespace lldb;
using namespace lldb_private;
#include "llvm/ADT/StringRef.h"

//-------------------------------------------------------------------------
// CommandObjectSettingsSet
//-------------------------------------------------------------------------

class CommandObjectSettingsSet : public CommandObjectRaw
{
public:
    CommandObjectSettingsSet (CommandInterpreter &interpreter) :
        CommandObjectRaw (interpreter,
                          "settings set",
                          "Set or change the value of a single debugger setting variable.",
                          NULL),
        m_options (interpreter)
    {
        CommandArgumentEntry arg1;
        CommandArgumentEntry arg2;
        CommandArgumentData var_name_arg;
        CommandArgumentData value_arg;

        // Define the first (and only) variant of this arg.
        var_name_arg.arg_type = eArgTypeSettingVariableName;
        var_name_arg.arg_repetition = eArgRepeatPlain;

        // There is only one variant this argument could be; put it into the argument entry.
        arg1.push_back (var_name_arg);

        // Define the first (and only) variant of this arg.
        value_arg.arg_type = eArgTypeValue;
        value_arg.arg_repetition = eArgRepeatPlain;

        // There is only one variant this argument could be; put it into the argument entry.
        arg2.push_back (value_arg);

        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg1);
        m_arguments.push_back (arg2);
        
        SetHelpLong (
"When setting a dictionary or array variable, you can set multiple entries \n\
at once by giving the values to the set command.  For example: \n\
\n\
(lldb) settings set target.run-args value1 value2 value3 \n\
(lldb) settings set target.env-vars MYPATH=~/.:/usr/bin  SOME_ENV_VAR=12345 \n\
\n\
(lldb) settings show target.run-args \n\
  [0]: 'value1' \n\
  [1]: 'value2' \n\
  [3]: 'value3' \n\
(lldb) settings show target.env-vars \n\
  'MYPATH=~/.:/usr/bin'\n\
  'SOME_ENV_VAR=12345' \n\
\n\
Warning:  The 'set' command re-sets the entire array or dictionary.  If you \n\
just want to add, remove or update individual values (or add something to \n\
the end), use one of the other settings sub-commands: append, replace, \n\
insert-before or insert-after.\n");

    }


    virtual
    ~CommandObjectSettingsSet () {}

    // Overrides base class's behavior where WantsCompletion = !WantsRawCommandString.
    virtual bool
    WantsCompletion() { return true; }

    virtual Options *
    GetOptions ()
    {
        return &m_options;
    }
    
    class CommandOptions : public Options
    {
    public:

        CommandOptions (CommandInterpreter &interpreter) :
            Options (interpreter),
            m_global (false)
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
                case 'g':
                    m_global = true;
                    break;
                default:
                    error.SetErrorStringWithFormat ("unrecognized options '%c'", short_option);
                    break;
            }

            return error;
        }

        void
        OptionParsingStarting ()
        {
            m_global = false;
        }
        
        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }

        // Options table: Required for subclasses of Options.

        static OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.

        bool m_global;
    };

    virtual int
    HandleArgumentCompletion (Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches)
    {
        std::string completion_str (input.GetArgumentAtIndex (cursor_index), cursor_char_position);

        const size_t argc = input.GetArgumentCount();
        const char *arg = NULL;
        int setting_var_idx;
        for (setting_var_idx = 1; setting_var_idx < argc; ++setting_var_idx)
        {
            arg = input.GetArgumentAtIndex(setting_var_idx);
            if (arg && arg[0] != '-')
                break; // We found our setting variable name index
        }
        if (cursor_index == setting_var_idx)
        {
            // Attempting to complete setting variable name
            CommandCompletions::InvokeCommonCompletionCallbacks (m_interpreter,
                                                                 CommandCompletions::eSettingsNameCompletion,
                                                                 completion_str.c_str(),
                                                                 match_start_point,
                                                                 max_return_elements,
                                                                 NULL,
                                                                 word_complete,
                                                                 matches);
        }
        else
        {
            arg = input.GetArgumentAtIndex(cursor_index);
            
            if (arg)
            {
                if (arg[0] == '-')
                {
                    // Complete option name
                }
                else
                {
                    // Complete setting value
                    const char *setting_var_name = input.GetArgumentAtIndex(setting_var_idx);
                    Error error;
                    lldb::OptionValueSP value_sp (m_interpreter.GetDebugger().GetPropertyValue(&m_exe_ctx, setting_var_name, false, error));
                    if (value_sp)
                    {
                        value_sp->AutoComplete (m_interpreter,
                                                completion_str.c_str(),
                                                match_start_point,
                                                max_return_elements,
                                                word_complete,
                                                matches);
                    }
                }
            }
        }
        return matches.GetSize();
    }
    
protected:
    virtual bool
    DoExecute (const char *command, CommandReturnObject &result)
    {
        Args cmd_args(command);

        // Process possible options.
        if (!ParseOptions (cmd_args, result))
            return false;

        const size_t argc = cmd_args.GetArgumentCount ();
        if ((argc < 2) && (!m_options.m_global))
        {
            result.AppendError ("'settings set' takes more arguments");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        const char *var_name = cmd_args.GetArgumentAtIndex (0);
        if ((var_name == NULL) || (var_name[0] == '\0'))
        {
            result.AppendError ("'settings set' command requires a valid variable name");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        // Split the raw command into var_name and value pair.
        llvm::StringRef raw_str(command);
        std::string var_value_string = raw_str.split(var_name).second.str();
        const char *var_value_cstr = Args::StripSpaces(var_value_string, true, true, false);

        Error error;
        if (m_options.m_global)
        {
            error = m_interpreter.GetDebugger().SetPropertyValue (NULL,
                                                                  eVarSetOperationAssign,
                                                                  var_name,
                                                                  var_value_cstr);
        }
        
        if (error.Success())
        {
            error = m_interpreter.GetDebugger().SetPropertyValue (&m_exe_ctx,
                                                                  eVarSetOperationAssign,
                                                                  var_name,
                                                                  var_value_cstr);
        }

        if (error.Fail())
        {
            result.AppendError (error.AsCString());
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        else
        {
            result.SetStatus (eReturnStatusSuccessFinishResult);
        }

        return result.Succeeded();
    }
private:
    CommandOptions m_options;
};

OptionDefinition
CommandObjectSettingsSet::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_2, false, "global", 'g', no_argument,   NULL, 0, eArgTypeNone, "Apply the new value to the global default value." },
    { 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};


//-------------------------------------------------------------------------
// CommandObjectSettingsShow -- Show current values
//-------------------------------------------------------------------------

class CommandObjectSettingsShow : public CommandObjectParsed
{
public:
    CommandObjectSettingsShow (CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter,
                             "settings show",
                             "Show the specified internal debugger setting variable and its value, or show all the currently set variables and their values, if nothing is specified.",
                             NULL)
    {
        CommandArgumentEntry arg1;
        CommandArgumentData var_name_arg;

        // Define the first (and only) variant of this arg.
        var_name_arg.arg_type = eArgTypeSettingVariableName;
        var_name_arg.arg_repetition = eArgRepeatOptional;

        // There is only one variant this argument could be; put it into the argument entry.
        arg1.push_back (var_name_arg);

        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg1);
    }

    virtual
    ~CommandObjectSettingsShow () {}


    virtual int
    HandleArgumentCompletion (Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches)
    {
        std::string completion_str (input.GetArgumentAtIndex (cursor_index), cursor_char_position);

        CommandCompletions::InvokeCommonCompletionCallbacks (m_interpreter,
                                                             CommandCompletions::eSettingsNameCompletion,
                                                             completion_str.c_str(),
                                                             match_start_point,
                                                             max_return_elements,
                                                             NULL,
                                                             word_complete,
                                                             matches);
        return matches.GetSize();
    }

protected:
    virtual bool
    DoExecute (Args& args, CommandReturnObject &result)
    {
        result.SetStatus (eReturnStatusSuccessFinishResult);

        const size_t argc = args.GetArgumentCount ();
        if (argc > 0)
        {
            for (size_t i=0; i<argc; ++i)
            {
                const char *property_path = args.GetArgumentAtIndex (i);

                Error error(m_interpreter.GetDebugger().DumpPropertyValue (&m_exe_ctx, result.GetOutputStream(), property_path, OptionValue::eDumpGroupValue));
                if (error.Success())
                {
                    result.GetOutputStream().EOL();
                }
                else
                {
                    result.AppendError (error.AsCString());
                    result.SetStatus (eReturnStatusFailed);
                }
            }
        }
        else
        {
            m_interpreter.GetDebugger().DumpAllPropertyValues (&m_exe_ctx, result.GetOutputStream(), OptionValue::eDumpGroupValue);
        }

        return result.Succeeded();
    }
};

//-------------------------------------------------------------------------
// CommandObjectSettingsList -- List settable variables
//-------------------------------------------------------------------------

class CommandObjectSettingsList : public CommandObjectParsed
{
public: 
    CommandObjectSettingsList (CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter,
                             "settings list",
                             "List and describe all the internal debugger settings variables that are available to the user to 'set' or 'show', or describe a particular variable or set of variables (by specifying the variable name or a common prefix).",
                             NULL)
    {
        CommandArgumentEntry arg;
        CommandArgumentData var_name_arg;
        CommandArgumentData prefix_name_arg;

        // Define the first variant of this arg.
        var_name_arg.arg_type = eArgTypeSettingVariableName;
        var_name_arg.arg_repetition = eArgRepeatOptional;

        // Define the second variant of this arg.
        prefix_name_arg.arg_type = eArgTypeSettingPrefix;
        prefix_name_arg.arg_repetition = eArgRepeatOptional;

        arg.push_back (var_name_arg);
        arg.push_back (prefix_name_arg);

        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg);
    }

    virtual
    ~CommandObjectSettingsList () {}

    virtual int
    HandleArgumentCompletion (Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches)
    {
        std::string completion_str (input.GetArgumentAtIndex (cursor_index), cursor_char_position);

        CommandCompletions::InvokeCommonCompletionCallbacks (m_interpreter,
                                                             CommandCompletions::eSettingsNameCompletion,
                                                             completion_str.c_str(),
                                                             match_start_point,
                                                             max_return_elements,
                                                             NULL,
                                                             word_complete,
                                                             matches);
        return matches.GetSize();
    }

protected:
    virtual bool
    DoExecute (Args& args, CommandReturnObject &result)
    {
        result.SetStatus (eReturnStatusSuccessFinishResult);

        const bool will_modify = false;
        const size_t argc = args.GetArgumentCount ();
        if (argc > 0)
        {
            const bool dump_qualified_name = true;

            for (size_t i=0; i<argc; ++i)
            {
                const char *property_path = args.GetArgumentAtIndex (i);
                
                const Property *property = m_interpreter.GetDebugger().GetValueProperties()->GetPropertyAtPath (&m_exe_ctx, will_modify, property_path);

                if (property)
                {
                    property->DumpDescription (m_interpreter, result.GetOutputStream(), 0, dump_qualified_name);
                }
                else
                {
                    result.AppendErrorWithFormat ("invalid property path '%s'", property_path);
                    result.SetStatus (eReturnStatusFailed);
                }
            }
        }
        else
        {
            m_interpreter.GetDebugger().DumpAllDescriptions (m_interpreter, result.GetOutputStream());
        }

        return result.Succeeded();
    }
};

//-------------------------------------------------------------------------
// CommandObjectSettingsRemove
//-------------------------------------------------------------------------

class CommandObjectSettingsRemove : public CommandObjectRaw
{
public:
    CommandObjectSettingsRemove (CommandInterpreter &interpreter) :
        CommandObjectRaw (interpreter,
                          "settings remove",
                          "Remove the specified element from an array or dictionary settings variable.",
                          NULL)
    {
        CommandArgumentEntry arg1;
        CommandArgumentEntry arg2;
        CommandArgumentData var_name_arg;
        CommandArgumentData index_arg;
        CommandArgumentData key_arg;

        // Define the first (and only) variant of this arg.
        var_name_arg.arg_type = eArgTypeSettingVariableName;
        var_name_arg.arg_repetition = eArgRepeatPlain;

        // There is only one variant this argument could be; put it into the argument entry.
        arg1.push_back (var_name_arg);

        // Define the first variant of this arg.
        index_arg.arg_type = eArgTypeSettingIndex;
        index_arg.arg_repetition = eArgRepeatPlain;

        // Define the second variant of this arg.
        key_arg.arg_type = eArgTypeSettingKey;
        key_arg.arg_repetition = eArgRepeatPlain;

        // Push both variants into this arg
        arg2.push_back (index_arg);
        arg2.push_back (key_arg);

        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg1);
        m_arguments.push_back (arg2);
    }

    virtual
    ~CommandObjectSettingsRemove () {}

    virtual int
    HandleArgumentCompletion (Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches)
    {
        std::string completion_str (input.GetArgumentAtIndex (cursor_index), cursor_char_position);

        // Attempting to complete variable name
        if (cursor_index < 2)
            CommandCompletions::InvokeCommonCompletionCallbacks (m_interpreter,
                                                                 CommandCompletions::eSettingsNameCompletion,
                                                                 completion_str.c_str(),
                                                                 match_start_point,
                                                                 max_return_elements,
                                                                 NULL,
                                                                 word_complete,
                                                                 matches);

        return matches.GetSize();
    }

protected:
    virtual bool
    DoExecute (const char *command, CommandReturnObject &result)
    {
        result.SetStatus (eReturnStatusSuccessFinishNoResult);
     
        Args cmd_args(command);
        
        // Process possible options.
        if (!ParseOptions (cmd_args, result))
            return false;
        
        const size_t argc = cmd_args.GetArgumentCount ();
        if (argc == 0)
        {
            result.AppendError ("'settings set' takes an array or dictionary item, or an array followed by one or more indexes, or a dictionary followed by one or more key names to remove");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        
        const char *var_name = cmd_args.GetArgumentAtIndex (0);
        if ((var_name == NULL) || (var_name[0] == '\0'))
        {
            result.AppendError ("'settings set' command requires a valid variable name");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        
        // Split the raw command into var_name and value pair.
        llvm::StringRef raw_str(command);
        std::string var_value_string = raw_str.split(var_name).second.str();
        const char *var_value_cstr = Args::StripSpaces(var_value_string, true, true, false);
        
        Error error (m_interpreter.GetDebugger().SetPropertyValue (&m_exe_ctx,
                                                                   eVarSetOperationRemove,
                                                                   var_name,
                                                                   var_value_cstr));
        if (error.Fail())
        {
            result.AppendError (error.AsCString());
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        
        return result.Succeeded();
    }
};

//-------------------------------------------------------------------------
// CommandObjectSettingsReplace
//-------------------------------------------------------------------------

class CommandObjectSettingsReplace : public CommandObjectRaw
{
public:
    CommandObjectSettingsReplace (CommandInterpreter &interpreter) :
        CommandObjectRaw (interpreter,
                          "settings replace",
                          "Replace the specified element from an internal debugger settings array or dictionary variable with the specified new value.",
                          NULL)
    {
        CommandArgumentEntry arg1;
        CommandArgumentEntry arg2;
        CommandArgumentEntry arg3;
        CommandArgumentData var_name_arg;
        CommandArgumentData index_arg;
        CommandArgumentData key_arg;
        CommandArgumentData value_arg;

        // Define the first (and only) variant of this arg.
        var_name_arg.arg_type = eArgTypeSettingVariableName;
        var_name_arg.arg_repetition = eArgRepeatPlain;

        // There is only one variant this argument could be; put it into the argument entry.
        arg1.push_back (var_name_arg);

        // Define the first (variant of this arg.
        index_arg.arg_type = eArgTypeSettingIndex;
        index_arg.arg_repetition = eArgRepeatPlain;

        // Define the second (variant of this arg.
        key_arg.arg_type = eArgTypeSettingKey;
        key_arg.arg_repetition = eArgRepeatPlain;

        // Put both variants into this arg
        arg2.push_back (index_arg);
        arg2.push_back (key_arg);

        // Define the first (and only) variant of this arg.
        value_arg.arg_type = eArgTypeValue;
        value_arg.arg_repetition = eArgRepeatPlain;

        // There is only one variant this argument could be; put it into the argument entry.
        arg3.push_back (value_arg);

        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg1);
        m_arguments.push_back (arg2);
        m_arguments.push_back (arg3);
    }


    virtual
    ~CommandObjectSettingsReplace () {}

    // Overrides base class's behavior where WantsCompletion = !WantsRawCommandString.
    virtual bool
    WantsCompletion() { return true; }

    virtual int
    HandleArgumentCompletion (Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches)
    {
        std::string completion_str (input.GetArgumentAtIndex (cursor_index), cursor_char_position);

        // Attempting to complete variable name
        if (cursor_index < 2)
            CommandCompletions::InvokeCommonCompletionCallbacks (m_interpreter,
                                                                 CommandCompletions::eSettingsNameCompletion,
                                                                 completion_str.c_str(),
                                                                 match_start_point,
                                                                 max_return_elements,
                                                                 NULL,
                                                                 word_complete,
                                                                 matches);

        return matches.GetSize();
    }

protected:
    virtual bool
    DoExecute (const char *command, CommandReturnObject &result)
    {
        result.SetStatus (eReturnStatusSuccessFinishNoResult);

        Args cmd_args(command);
        const char *var_name = cmd_args.GetArgumentAtIndex (0);
        if ((var_name == NULL) || (var_name[0] == '\0'))
        {
            result.AppendError ("'settings replace' command requires a valid variable name; No value supplied");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }


        // Split the raw command into var_name, index_value, and value triple.
        llvm::StringRef raw_str(command);
        std::string var_value_string = raw_str.split(var_name).second.str();
        const char *var_value_cstr = Args::StripSpaces(var_value_string, true, true, false);

        Error error(m_interpreter.GetDebugger().SetPropertyValue (&m_exe_ctx,
                                                                  eVarSetOperationReplace,
                                                                  var_name,
                                                                  var_value_cstr));
        if (error.Fail())
        {
            result.AppendError (error.AsCString());
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        else
        {
            result.SetStatus (eReturnStatusSuccessFinishNoResult);

        }

        return result.Succeeded();
    }
};

//-------------------------------------------------------------------------
// CommandObjectSettingsInsertBefore
//-------------------------------------------------------------------------

class CommandObjectSettingsInsertBefore : public CommandObjectRaw
{
public:
    CommandObjectSettingsInsertBefore (CommandInterpreter &interpreter) :
        CommandObjectRaw (interpreter,
                          "settings insert-before",
                          "Insert value(s) into an internal debugger settings array variable, immediately before the specified element.",
                          NULL)
    {
        CommandArgumentEntry arg1;
        CommandArgumentEntry arg2;
        CommandArgumentEntry arg3;
        CommandArgumentData var_name_arg;
        CommandArgumentData index_arg;
        CommandArgumentData value_arg;

        // Define the first (and only) variant of this arg.
        var_name_arg.arg_type = eArgTypeSettingVariableName;
        var_name_arg.arg_repetition = eArgRepeatPlain;

        // There is only one variant this argument could be; put it into the argument entry.
        arg1.push_back (var_name_arg);

        // Define the first (variant of this arg.
        index_arg.arg_type = eArgTypeSettingIndex;
        index_arg.arg_repetition = eArgRepeatPlain;

        // There is only one variant this argument could be; put it into the argument entry.
        arg2.push_back (index_arg);

        // Define the first (and only) variant of this arg.
        value_arg.arg_type = eArgTypeValue;
        value_arg.arg_repetition = eArgRepeatPlain;

        // There is only one variant this argument could be; put it into the argument entry.
        arg3.push_back (value_arg);

        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg1);
        m_arguments.push_back (arg2);
        m_arguments.push_back (arg3);
    }

    virtual
    ~CommandObjectSettingsInsertBefore () {}

    // Overrides base class's behavior where WantsCompletion = !WantsRawCommandString.
    virtual bool
    WantsCompletion() { return true; }

    virtual int
    HandleArgumentCompletion (Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches)
    {
        std::string completion_str (input.GetArgumentAtIndex (cursor_index), cursor_char_position);

        // Attempting to complete variable name
        if (cursor_index < 2)
            CommandCompletions::InvokeCommonCompletionCallbacks (m_interpreter,
                                                                 CommandCompletions::eSettingsNameCompletion,
                                                                 completion_str.c_str(),
                                                                 match_start_point,
                                                                 max_return_elements,
                                                                 NULL,
                                                                 word_complete,
                                                                 matches);

        return matches.GetSize();
    }

protected:
    virtual bool
    DoExecute (const char *command, CommandReturnObject &result)
    {
        result.SetStatus (eReturnStatusSuccessFinishNoResult);

        Args cmd_args(command);
        const size_t argc = cmd_args.GetArgumentCount ();

        if (argc < 3)
        {
            result.AppendError ("'settings insert-before' takes more arguments");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        const char *var_name = cmd_args.GetArgumentAtIndex (0);
        if ((var_name == NULL) || (var_name[0] == '\0'))
        {
            result.AppendError ("'settings insert-before' command requires a valid variable name; No value supplied");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        // Split the raw command into var_name, index_value, and value triple.
        llvm::StringRef raw_str(command);
        std::string var_value_string = raw_str.split(var_name).second.str();
        const char *var_value_cstr = Args::StripSpaces(var_value_string, true, true, false);

        Error error(m_interpreter.GetDebugger().SetPropertyValue (&m_exe_ctx,
                                                                  eVarSetOperationInsertBefore,
                                                                  var_name,
                                                                  var_value_cstr));
        if (error.Fail())
        {
            result.AppendError (error.AsCString());
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        return result.Succeeded();
    }
};

//-------------------------------------------------------------------------
// CommandObjectSettingInsertAfter
//-------------------------------------------------------------------------

class CommandObjectSettingsInsertAfter : public CommandObjectRaw
{
public:
    CommandObjectSettingsInsertAfter (CommandInterpreter &interpreter) :
        CommandObjectRaw (interpreter,
                          "settings insert-after",
                          "Insert value(s) into an internal debugger settings array variable, immediately after the specified element.",
                          NULL)
    {
        CommandArgumentEntry arg1;
        CommandArgumentEntry arg2;
        CommandArgumentEntry arg3;
        CommandArgumentData var_name_arg;
        CommandArgumentData index_arg;
        CommandArgumentData value_arg;

        // Define the first (and only) variant of this arg.
        var_name_arg.arg_type = eArgTypeSettingVariableName;
        var_name_arg.arg_repetition = eArgRepeatPlain;

        // There is only one variant this argument could be; put it into the argument entry.
        arg1.push_back (var_name_arg);

        // Define the first (variant of this arg.
        index_arg.arg_type = eArgTypeSettingIndex;
        index_arg.arg_repetition = eArgRepeatPlain;

        // There is only one variant this argument could be; put it into the argument entry.
        arg2.push_back (index_arg);

        // Define the first (and only) variant of this arg.
        value_arg.arg_type = eArgTypeValue;
        value_arg.arg_repetition = eArgRepeatPlain;

        // There is only one variant this argument could be; put it into the argument entry.
        arg3.push_back (value_arg);

        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg1);
        m_arguments.push_back (arg2);
        m_arguments.push_back (arg3);
    }

    virtual
    ~CommandObjectSettingsInsertAfter () {}

    // Overrides base class's behavior where WantsCompletion = !WantsRawCommandString.
    virtual bool
    WantsCompletion() { return true; }

    virtual int
    HandleArgumentCompletion (Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches)
    {
        std::string completion_str (input.GetArgumentAtIndex (cursor_index), cursor_char_position);

        // Attempting to complete variable name
        if (cursor_index < 2)
            CommandCompletions::InvokeCommonCompletionCallbacks (m_interpreter,
                                                                 CommandCompletions::eSettingsNameCompletion,
                                                                 completion_str.c_str(),
                                                                 match_start_point,
                                                                 max_return_elements,
                                                                 NULL,
                                                                 word_complete,
                                                                 matches);

        return matches.GetSize();
    }
    
protected:
    virtual bool
    DoExecute (const char *command, CommandReturnObject &result)
    {
        result.SetStatus (eReturnStatusSuccessFinishNoResult);

        Args cmd_args(command);
        const size_t argc = cmd_args.GetArgumentCount ();

        if (argc < 3)
        {
            result.AppendError ("'settings insert-after' takes more arguments");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        const char *var_name = cmd_args.GetArgumentAtIndex (0);
        if ((var_name == NULL) || (var_name[0] == '\0'))
        {
            result.AppendError ("'settings insert-after' command requires a valid variable name; No value supplied");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        // Split the raw command into var_name, index_value, and value triple.
        llvm::StringRef raw_str(command);
        std::string var_value_string = raw_str.split(var_name).second.str();
        const char *var_value_cstr = Args::StripSpaces(var_value_string, true, true, false);

        Error error(m_interpreter.GetDebugger().SetPropertyValue (&m_exe_ctx,
                                                                  eVarSetOperationInsertAfter,
                                                                  var_name,
                                                                  var_value_cstr));
        if (error.Fail())
        {
            result.AppendError (error.AsCString());
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        return result.Succeeded();
    }
};

//-------------------------------------------------------------------------
// CommandObjectSettingsAppend
//-------------------------------------------------------------------------

class CommandObjectSettingsAppend : public CommandObjectRaw
{
public:
    CommandObjectSettingsAppend (CommandInterpreter &interpreter) :
        CommandObjectRaw (interpreter,
                          "settings append",
                          "Append a new value to the end of an internal debugger settings array, dictionary or string variable.",
                          NULL)
    {
        CommandArgumentEntry arg1;
        CommandArgumentEntry arg2;
        CommandArgumentData var_name_arg;
        CommandArgumentData value_arg;

        // Define the first (and only) variant of this arg.
        var_name_arg.arg_type = eArgTypeSettingVariableName;
        var_name_arg.arg_repetition = eArgRepeatPlain;

        // There is only one variant this argument could be; put it into the argument entry.
        arg1.push_back (var_name_arg);

        // Define the first (and only) variant of this arg.
        value_arg.arg_type = eArgTypeValue;
        value_arg.arg_repetition = eArgRepeatPlain;

        // There is only one variant this argument could be; put it into the argument entry.
        arg2.push_back (value_arg);

        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg1);
        m_arguments.push_back (arg2);
    }

    virtual
    ~CommandObjectSettingsAppend () {}

    // Overrides base class's behavior where WantsCompletion = !WantsRawCommandString.
    virtual bool
    WantsCompletion() { return true; }

    virtual int
    HandleArgumentCompletion (Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches)
    {
        std::string completion_str (input.GetArgumentAtIndex (cursor_index), cursor_char_position);

        // Attempting to complete variable name
        if (cursor_index < 2)
            CommandCompletions::InvokeCommonCompletionCallbacks (m_interpreter,
                                                                 CommandCompletions::eSettingsNameCompletion,
                                                                 completion_str.c_str(),
                                                                 match_start_point,
                                                                 max_return_elements,
                                                                 NULL,
                                                                 word_complete,
                                                                 matches);

        return matches.GetSize();
    }

protected:
    virtual bool
    DoExecute (const char *command, CommandReturnObject &result)
    {
        result.SetStatus (eReturnStatusSuccessFinishNoResult);
        Args cmd_args(command);
        const size_t argc = cmd_args.GetArgumentCount ();

        if (argc < 2)
        {
            result.AppendError ("'settings append' takes more arguments");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        const char *var_name = cmd_args.GetArgumentAtIndex (0);
        if ((var_name == NULL) || (var_name[0] == '\0'))
        {
            result.AppendError ("'settings append' command requires a valid variable name; No value supplied");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        // Do not perform cmd_args.Shift() since StringRef is manipulating the
        // raw character string later on.

        // Split the raw command into var_name and value pair.
        llvm::StringRef raw_str(command);
        std::string var_value_string = raw_str.split(var_name).second.str();
        const char *var_value_cstr = Args::StripSpaces(var_value_string, true, true, false);

        Error error(m_interpreter.GetDebugger().SetPropertyValue (&m_exe_ctx,
                                                                  eVarSetOperationAppend,
                                                                  var_name,
                                                                  var_value_cstr));
        if (error.Fail())
        {
            result.AppendError (error.AsCString());
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        return result.Succeeded();
    }
};

//-------------------------------------------------------------------------
// CommandObjectSettingsClear
//-------------------------------------------------------------------------

class CommandObjectSettingsClear : public CommandObjectParsed
{
public:
    CommandObjectSettingsClear (CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter,
                             "settings clear",
                             "Erase all the contents of an internal debugger settings variables; this is only valid for variables with clearable types, i.e. strings, arrays or dictionaries.",
                             NULL)
    {
        CommandArgumentEntry arg;
        CommandArgumentData var_name_arg;

        // Define the first (and only) variant of this arg.
        var_name_arg.arg_type = eArgTypeSettingVariableName;
        var_name_arg.arg_repetition = eArgRepeatPlain;

        // There is only one variant this argument could be; put it into the argument entry.
        arg.push_back (var_name_arg);

        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg);
    }

    virtual
    ~CommandObjectSettingsClear () {}

    virtual int
    HandleArgumentCompletion (Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches)
    {
        std::string completion_str (input.GetArgumentAtIndex (cursor_index), cursor_char_position);

        // Attempting to complete variable name
        if (cursor_index < 2)
            CommandCompletions::InvokeCommonCompletionCallbacks (m_interpreter,
                                                                 CommandCompletions::eSettingsNameCompletion,
                                                                 completion_str.c_str(),
                                                                 match_start_point,
                                                                 max_return_elements,
                                                                 NULL,
                                                                 word_complete,
                                                                 matches);

        return matches.GetSize();
    }

protected:
    virtual bool
    DoExecute (Args& command, CommandReturnObject &result)
    {
        result.SetStatus (eReturnStatusSuccessFinishNoResult);
        const size_t argc = command.GetArgumentCount ();

        if (argc != 1)
        {
            result.AppendError ("'setttings clear' takes exactly one argument");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        const char *var_name = command.GetArgumentAtIndex (0);
        if ((var_name == NULL) || (var_name[0] == '\0'))
        {
            result.AppendError ("'settings clear' command requires a valid variable name; No value supplied");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        
        Error error (m_interpreter.GetDebugger().SetPropertyValue (&m_exe_ctx,
                                                                   eVarSetOperationClear,
                                                                   var_name,
                                                                   NULL));
        if (error.Fail())
        {
            result.AppendError (error.AsCString());
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        return result.Succeeded();
    }
};

//-------------------------------------------------------------------------
// CommandObjectMultiwordSettings
//-------------------------------------------------------------------------

CommandObjectMultiwordSettings::CommandObjectMultiwordSettings (CommandInterpreter &interpreter) :
    CommandObjectMultiword (interpreter,
                            "settings",
                            "A set of commands for manipulating internal settable debugger variables.",
                            "settings <command> [<command-options>]")
{
    LoadSubCommand ("set",           CommandObjectSP (new CommandObjectSettingsSet (interpreter)));
    LoadSubCommand ("show",          CommandObjectSP (new CommandObjectSettingsShow (interpreter)));
    LoadSubCommand ("list",          CommandObjectSP (new CommandObjectSettingsList (interpreter)));
    LoadSubCommand ("remove",        CommandObjectSP (new CommandObjectSettingsRemove (interpreter)));
    LoadSubCommand ("replace",       CommandObjectSP (new CommandObjectSettingsReplace (interpreter)));
    LoadSubCommand ("insert-before", CommandObjectSP (new CommandObjectSettingsInsertBefore (interpreter)));
    LoadSubCommand ("insert-after",  CommandObjectSP (new CommandObjectSettingsInsertAfter (interpreter)));
    LoadSubCommand ("append",        CommandObjectSP (new CommandObjectSettingsAppend (interpreter)));
    LoadSubCommand ("clear",         CommandObjectSP (new CommandObjectSettingsClear (interpreter)));
}

CommandObjectMultiwordSettings::~CommandObjectMultiwordSettings ()
{
}
