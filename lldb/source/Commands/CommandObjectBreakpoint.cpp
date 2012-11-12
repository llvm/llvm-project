//===-- CommandObjectBreakpoint.cpp -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectBreakpoint.h"
#include "CommandObjectBreakpointCommand.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Breakpoint/Breakpoint.h"
#include "lldb/Breakpoint/BreakpointIDList.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Target/Target.h"
#include "lldb/Interpreter/CommandCompletions.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadSpec.h"

#include <vector>

using namespace lldb;
using namespace lldb_private;

static void
AddBreakpointDescription (Stream *s, Breakpoint *bp, lldb::DescriptionLevel level)
{
    s->IndentMore();
    bp->GetDescription (s, level, true);
    s->IndentLess();
    s->EOL();
}

//-------------------------------------------------------------------------
// CommandObjectBreakpointSet
//-------------------------------------------------------------------------


class CommandObjectBreakpointSet : public CommandObjectParsed
{
public:

    typedef enum BreakpointSetType
    {
        eSetTypeInvalid,
        eSetTypeFileAndLine,
        eSetTypeAddress,
        eSetTypeFunctionName,
        eSetTypeFunctionRegexp,
        eSetTypeSourceRegexp,
        eSetTypeException
    } BreakpointSetType;

    CommandObjectBreakpointSet (CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter,
                             "breakpoint set", 
                             "Sets a breakpoint or set of breakpoints in the executable.", 
                             "breakpoint set <cmd-options>"),
        m_options (interpreter)
    {
    }


    virtual
    ~CommandObjectBreakpointSet () {}

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
            m_condition (),
            m_filenames (),
            m_line_num (0),
            m_column (0),
            m_func_names (),
            m_func_name_type_mask (eFunctionNameTypeNone),
            m_func_regexp (),
            m_source_text_regexp(),
            m_modules (),
            m_load_addr(),
            m_ignore_count (0),
            m_thread_id(LLDB_INVALID_THREAD_ID),
            m_thread_index (UINT32_MAX),
            m_thread_name(),
            m_queue_name(),
            m_catch_bp (false),
            m_throw_bp (true),
            m_language (eLanguageTypeUnknown),
            m_skip_prologue (eLazyBoolCalculate),
            m_one_shot (false)
        {
        }


        virtual
        ~CommandOptions () {}

        virtual Error
        SetOptionValue (uint32_t option_idx, const char *option_arg)
        {
            Error error;
            char short_option = (char) m_getopt_table[option_idx].val;

            switch (short_option)
            {
                case 'a':
                    m_load_addr = Args::StringToUInt64(option_arg, LLDB_INVALID_ADDRESS, 0);
                    if (m_load_addr == LLDB_INVALID_ADDRESS)
                        m_load_addr = Args::StringToUInt64(option_arg, LLDB_INVALID_ADDRESS, 16);

                    if (m_load_addr == LLDB_INVALID_ADDRESS)
                        error.SetErrorStringWithFormat ("invalid address string '%s'", option_arg);
                    break;

                case 'b':
                    m_func_names.push_back (option_arg);
                    m_func_name_type_mask |= eFunctionNameTypeBase;
                    break;

                case 'C':
                    m_column = Args::StringToUInt32 (option_arg, 0);
                    break;

                case 'c':
                    m_condition.assign(option_arg);
                    break;

                case 'E':
                {
                    LanguageType language = LanguageRuntime::GetLanguageTypeFromString (option_arg);

                    switch (language)
                    {
                        case eLanguageTypeC89:
                        case eLanguageTypeC:
                        case eLanguageTypeC99:
                            m_language = eLanguageTypeC;
                            break;
                        case eLanguageTypeC_plus_plus:
                            m_language = eLanguageTypeC_plus_plus;
                            break;
                        case eLanguageTypeObjC:
                            m_language = eLanguageTypeObjC;
                            break;
                        case eLanguageTypeObjC_plus_plus:
                            error.SetErrorStringWithFormat ("Set exception breakpoints separately for c++ and objective-c");
                            break;
                        case eLanguageTypeUnknown:
                            error.SetErrorStringWithFormat ("Unknown language type: '%s' for exception breakpoint", option_arg);
                            break;
                        default:
                            error.SetErrorStringWithFormat ("Unsupported language type: '%s' for exception breakpoint", option_arg);
                    }
                }
                break;

                case 'f':
                    m_filenames.AppendIfUnique (FileSpec(option_arg, false));
                    break;

                case 'F':
                    m_func_names.push_back (option_arg);
                    m_func_name_type_mask |= eFunctionNameTypeFull;
                    break;

                case 'h':
                {
                    bool success;
                    m_catch_bp = Args::StringToBoolean (option_arg, true, &success);
                    if (!success)
                        error.SetErrorStringWithFormat ("Invalid boolean value for on-catch option: '%s'", option_arg);
                }

                case 'i':
                {
                    m_ignore_count = Args::StringToUInt32(option_arg, UINT32_MAX, 0);
                    if (m_ignore_count == UINT32_MAX)
                       error.SetErrorStringWithFormat ("invalid ignore count '%s'", option_arg);
                    break;
                }

                case 'K':
                {
                    bool success;
                    bool value;
                    value = Args::StringToBoolean (option_arg, true, &success);
                    if (value)
                        m_skip_prologue = eLazyBoolYes;
                    else
                        m_skip_prologue = eLazyBoolNo;
                        
                    if (!success)
                        error.SetErrorStringWithFormat ("Invalid boolean value for skip prologue option: '%s'", option_arg);
                }
                break;

                case 'l':
                    m_line_num = Args::StringToUInt32 (option_arg, 0);
                    break;

                case 'M':
                    m_func_names.push_back (option_arg);
                    m_func_name_type_mask |= eFunctionNameTypeMethod;
                    break;

                case 'n':
                    m_func_names.push_back (option_arg);
                    m_func_name_type_mask |= eFunctionNameTypeAuto;
                    break;

                case 'o':
                    m_one_shot = true;
                    break;

                case 'p':
                    m_source_text_regexp.assign (option_arg);
                    break;
                    
                case 'q':
                    m_queue_name.assign (option_arg);
                    break;

                case 'r':
                    m_func_regexp.assign (option_arg);
                    break;

                case 's':
                {
                    m_modules.AppendIfUnique (FileSpec (option_arg, false));
                    break;
                }
                    
                case 'S':
                    m_func_names.push_back (option_arg);
                    m_func_name_type_mask |= eFunctionNameTypeSelector;
                    break;

                case 't' :
                {
                    m_thread_id = Args::StringToUInt64(option_arg, LLDB_INVALID_THREAD_ID, 0);
                    if (m_thread_id == LLDB_INVALID_THREAD_ID)
                       error.SetErrorStringWithFormat ("invalid thread id string '%s'", option_arg);
                }
                break;

                case 'T':
                    m_thread_name.assign (option_arg);
                    break;

                case 'w':
                {
                    bool success;
                    m_throw_bp = Args::StringToBoolean (option_arg, true, &success);
                    if (!success)
                        error.SetErrorStringWithFormat ("Invalid boolean value for on-throw option: '%s'", option_arg);
                }
                break;

                case 'x':
                {
                    m_thread_index = Args::StringToUInt32(option_arg, UINT32_MAX, 0);
                    if (m_thread_id == UINT32_MAX)
                       error.SetErrorStringWithFormat ("invalid thread index string '%s'", option_arg);
                    
                }
                break;

                default:
                    error.SetErrorStringWithFormat ("unrecognized option '%c'", short_option);
                    break;
            }

            return error;
        }
        void
        OptionParsingStarting ()
        {
            m_condition.clear();
            m_filenames.Clear();
            m_line_num = 0;
            m_column = 0;
            m_func_names.clear();
            m_func_name_type_mask = eFunctionNameTypeNone;
            m_func_regexp.clear();
            m_source_text_regexp.clear();
            m_modules.Clear();
            m_load_addr = LLDB_INVALID_ADDRESS;
            m_ignore_count = 0;
            m_thread_id = LLDB_INVALID_THREAD_ID;
            m_thread_index = UINT32_MAX;
            m_thread_name.clear();
            m_queue_name.clear();
            m_catch_bp = false;
            m_throw_bp = true;
            m_language = eLanguageTypeUnknown;
            m_skip_prologue = eLazyBoolCalculate;
            m_one_shot = false;
        }
    
        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }

        // Options table: Required for subclasses of Options.

        static OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.

        std::string m_condition;
        FileSpecList m_filenames;
        uint32_t m_line_num;
        uint32_t m_column;
        std::vector<std::string> m_func_names;
        uint32_t m_func_name_type_mask;
        std::string m_func_regexp;
        std::string m_source_text_regexp;
        FileSpecList m_modules;
        lldb::addr_t m_load_addr;
        uint32_t m_ignore_count;
        lldb::tid_t m_thread_id;
        uint32_t m_thread_index;
        std::string m_thread_name;
        std::string m_queue_name;
        bool m_catch_bp;
        bool m_throw_bp;
        lldb::LanguageType m_language;
        LazyBool m_skip_prologue;
        bool m_one_shot;

    };

protected:
    virtual bool
    DoExecute (Args& command,
             CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target == NULL)
        {
            result.AppendError ("Invalid target.  Must set target before setting breakpoints (see 'target create' command).");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        // The following are the various types of breakpoints that could be set:
        //   1).  -f -l -p  [-s -g]   (setting breakpoint by source location)
        //   2).  -a  [-s -g]         (setting breakpoint by address)
        //   3).  -n  [-s -g]         (setting breakpoint by function name)
        //   4).  -r  [-s -g]         (setting breakpoint by function name regular expression)
        //   5).  -p -f               (setting a breakpoint by comparing a reg-exp to source text)
        //   6).  -E [-w -h]          (setting a breakpoint for exceptions for a given language.)

        BreakpointSetType break_type = eSetTypeInvalid;

        if (m_options.m_line_num != 0)
            break_type = eSetTypeFileAndLine;
        else if (m_options.m_load_addr != LLDB_INVALID_ADDRESS)
            break_type = eSetTypeAddress;
        else if (!m_options.m_func_names.empty())
            break_type = eSetTypeFunctionName;
        else if  (!m_options.m_func_regexp.empty())
            break_type = eSetTypeFunctionRegexp;
        else if (!m_options.m_source_text_regexp.empty())
            break_type = eSetTypeSourceRegexp;
        else if (m_options.m_language != eLanguageTypeUnknown)
            break_type = eSetTypeException;

        Breakpoint *bp = NULL;
        FileSpec module_spec;
        const bool internal = false;

        switch (break_type)
        {
            case eSetTypeFileAndLine: // Breakpoint by source position
                {
                    FileSpec file;
                    uint32_t num_files = m_options.m_filenames.GetSize();
                    if (num_files == 0)
                    {
                        if (!GetDefaultFile (target, file, result))
                        {
                            result.AppendError("No file supplied and no default file available.");
                            result.SetStatus (eReturnStatusFailed);
                            return false;
                        }
                    }
                    else if (num_files > 1)
                    {
                        result.AppendError("Only one file at a time is allowed for file and line breakpoints.");
                        result.SetStatus (eReturnStatusFailed);
                        return false;
                    }
                    else
                        file = m_options.m_filenames.GetFileSpecAtIndex(0);
                    
                    // Only check for inline functions if 
                    LazyBool check_inlines = eLazyBoolCalculate;
                    
                    bp = target->CreateBreakpoint (&(m_options.m_modules),
                                                   file,
                                                   m_options.m_line_num,
                                                   check_inlines,
                                                   m_options.m_skip_prologue,
                                                   internal).get();
                }
                break;

            case eSetTypeAddress: // Breakpoint by address
                bp = target->CreateBreakpoint (m_options.m_load_addr, false).get();
                break;

            case eSetTypeFunctionName: // Breakpoint by function name
                {
                    uint32_t name_type_mask = m_options.m_func_name_type_mask;
                    
                    if (name_type_mask == 0)
                        name_type_mask = eFunctionNameTypeAuto;
                    
                    bp = target->CreateBreakpoint (&(m_options.m_modules),
                                                   &(m_options.m_filenames),
                                                   m_options.m_func_names,
                                                   name_type_mask,
                                                   m_options.m_skip_prologue,
                                                   internal).get();
                }
                break;

            case eSetTypeFunctionRegexp: // Breakpoint by regular expression function name
                {
                    RegularExpression regexp(m_options.m_func_regexp.c_str());
                    if (!regexp.IsValid())
                    {
                        char err_str[1024];
                        regexp.GetErrorAsCString(err_str, sizeof(err_str));
                        result.AppendErrorWithFormat("Function name regular expression could not be compiled: \"%s\"",
                                                     err_str);
                        result.SetStatus (eReturnStatusFailed);
                        return false;
                    }
                    
                    bp = target->CreateFuncRegexBreakpoint (&(m_options.m_modules),
                                                            &(m_options.m_filenames),
                                                            regexp,
                                                            m_options.m_skip_prologue,
                                                            internal).get();
                }
                break;
            case eSetTypeSourceRegexp: // Breakpoint by regexp on source text.
                {
                    int num_files = m_options.m_filenames.GetSize();
                    
                    if (num_files == 0)
                    {
                        FileSpec file;
                        if (!GetDefaultFile (target, file, result))
                        {
                            result.AppendError ("No files provided and could not find default file.");
                            result.SetStatus (eReturnStatusFailed);
                            return false;
                        }
                        else
                        {
                            m_options.m_filenames.Append (file);
                        }
                    }
                    
                    RegularExpression regexp(m_options.m_source_text_regexp.c_str());
                    if (!regexp.IsValid())
                    {
                        char err_str[1024];
                        regexp.GetErrorAsCString(err_str, sizeof(err_str));
                        result.AppendErrorWithFormat("Source text regular expression could not be compiled: \"%s\"",
                                                     err_str);
                        result.SetStatus (eReturnStatusFailed);
                        return false;
                    }
                    bp = target->CreateSourceRegexBreakpoint (&(m_options.m_modules), &(m_options.m_filenames), regexp).get();
                }
                break;
            case eSetTypeException:
                {
                    bp = target->CreateExceptionBreakpoint (m_options.m_language, m_options.m_catch_bp, m_options.m_throw_bp).get();
                }
                break;
            default:
                break;
        }

        // Now set the various options that were passed in:
        if (bp)
        {
            if (m_options.m_thread_id != LLDB_INVALID_THREAD_ID)
                bp->SetThreadID (m_options.m_thread_id);
                
            if (m_options.m_thread_index != UINT32_MAX)
                bp->GetOptions()->GetThreadSpec()->SetIndex(m_options.m_thread_index);
            
            if (!m_options.m_thread_name.empty())
                bp->GetOptions()->GetThreadSpec()->SetName(m_options.m_thread_name.c_str());
            
            if (!m_options.m_queue_name.empty())
                bp->GetOptions()->GetThreadSpec()->SetQueueName(m_options.m_queue_name.c_str());
                
            if (m_options.m_ignore_count != 0)
                bp->GetOptions()->SetIgnoreCount(m_options.m_ignore_count);

            if (!m_options.m_condition.empty())
                bp->GetOptions()->SetCondition(m_options.m_condition.c_str());
            
            bp->SetOneShot (m_options.m_one_shot);
        }
        
        if (bp)
        {
            Stream &output_stream = result.GetOutputStream();
            const bool show_locations = false;
            bp->GetDescription(&output_stream, lldb::eDescriptionLevelInitial, show_locations);
            // Don't print out this warning for exception breakpoints.  They can get set before the target
            // is set, but we won't know how to actually set the breakpoint till we run.
            if (bp->GetNumLocations() == 0 && break_type != eSetTypeException)
                output_stream.Printf ("WARNING:  Unable to resolve breakpoint to any actual locations.\n");
            result.SetStatus (eReturnStatusSuccessFinishResult);
        }
        else if (!bp)
        {
            result.AppendError ("Breakpoint creation failed: No breakpoint created.");
            result.SetStatus (eReturnStatusFailed);
        }

        return result.Succeeded();
    }

private:
    bool
    GetDefaultFile (Target *target, FileSpec &file, CommandReturnObject &result)
    {
        uint32_t default_line;
        // First use the Source Manager's default file. 
        // Then use the current stack frame's file.
        if (!target->GetSourceManager().GetDefaultFileAndLine(file, default_line))
        {
            StackFrame *cur_frame = m_interpreter.GetExecutionContext().GetFramePtr();
            if (cur_frame == NULL)
            {
                result.AppendError ("No selected frame to use to find the default file.");
                result.SetStatus (eReturnStatusFailed);
                return false;
            }
            else if (!cur_frame->HasDebugInformation())
            {
                result.AppendError ("Cannot use the selected frame to find the default file, it has no debug info.");
                result.SetStatus (eReturnStatusFailed);
                return false;
            }
            else
            {
                const SymbolContext &sc = cur_frame->GetSymbolContext (eSymbolContextLineEntry);
                if (sc.line_entry.file)
                {
                    file = sc.line_entry.file;
                }
                else
                {
                    result.AppendError ("Can't find the file for the selected frame to use as the default file.");
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
            }
        }
        return true;
    }
    
    CommandOptions m_options;
};
// If an additional option set beyond LLDB_OPTION_SET_10 is added, make sure to
// update the numbers passed to LLDB_OPT_SET_FROM_TO(...) appropriately.
#define LLDB_OPT_FILE ( LLDB_OPT_SET_FROM_TO(1, 9) & ~LLDB_OPT_SET_2 )
#define LLDB_OPT_NOT_10 ( LLDB_OPT_SET_FROM_TO(1, 10) & ~LLDB_OPT_SET_10 )
#define LLDB_OPT_SKIP_PROLOGUE ( LLDB_OPT_SET_1 | LLDB_OPT_SET_FROM_TO(3,8) )

OptionDefinition
CommandObjectBreakpointSet::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_NOT_10, false, "shlib", 's', required_argument, NULL, CommandCompletions::eModuleCompletion, eArgTypeShlibName,
        "Set the breakpoint only in this shared library.  "
        "Can repeat this option multiple times to specify multiple shared libraries."},

    { LLDB_OPT_SET_ALL, false, "ignore-count", 'i', required_argument,   NULL, 0, eArgTypeCount,
        "Set the number of times this breakpoint is skipped before stopping." },

    { LLDB_OPT_SET_ALL, false, "one-shot", 'o', no_argument,   NULL, 0, eArgTypeNone,
        "The breakpoint is deleted the first time it stop causes a stop." },

    { LLDB_OPT_SET_ALL, false, "condition",    'c', required_argument, NULL, 0, eArgTypeExpression,
        "The breakpoint stops only if this condition expression evaluates to true."},

    { LLDB_OPT_SET_ALL, false, "thread-index", 'x', required_argument, NULL, 0, eArgTypeThreadIndex,
        "The breakpoint stops only for the thread whose index matches this argument."},

    { LLDB_OPT_SET_ALL, false, "thread-id", 't', required_argument, NULL, 0, eArgTypeThreadID,
        "The breakpoint stops only for the thread whose TID matches this argument."},

    { LLDB_OPT_SET_ALL, false, "thread-name", 'T', required_argument, NULL, 0, eArgTypeThreadName,
        "The breakpoint stops only for the thread whose thread name matches this argument."},

    { LLDB_OPT_SET_ALL, false, "queue-name", 'q', required_argument, NULL, 0, eArgTypeQueueName,
        "The breakpoint stops only for threads in the queue whose name is given by this argument."},

    { LLDB_OPT_FILE, false, "file", 'f', required_argument, NULL, CommandCompletions::eSourceFileCompletion, eArgTypeFilename,
        "Specifies the source file in which to set this breakpoint."},

    { LLDB_OPT_SET_1, true, "line", 'l', required_argument, NULL, 0, eArgTypeLineNum,
        "Specifies the line number on which to set this breakpoint."},

    // Comment out this option for the moment, as we don't actually use it, but will in the future.
    // This way users won't see it, but the infrastructure is left in place.
    //    { 0, false, "column",     'C', required_argument, NULL, "<column>",
    //    "Set the breakpoint by source location at this particular column."},

    { LLDB_OPT_SET_2, true, "address", 'a', required_argument, NULL, 0, eArgTypeAddress,
        "Set the breakpoint by address, at the specified address."},

    { LLDB_OPT_SET_3, true, "name", 'n', required_argument, NULL, CommandCompletions::eSymbolCompletion, eArgTypeFunctionName,
        "Set the breakpoint by function name.  Can be repeated multiple times to make one breakpoint for multiple snames" },

    { LLDB_OPT_SET_4, true, "fullname", 'F', required_argument, NULL, CommandCompletions::eSymbolCompletion, eArgTypeFullName,
        "Set the breakpoint by fully qualified function names. For C++ this means namespaces and all arguments, and "
        "for Objective C this means a full function prototype with class and selector.   "
        "Can be repeated multiple times to make one breakpoint for multiple names." },

    { LLDB_OPT_SET_5, true, "selector", 'S', required_argument, NULL, 0, eArgTypeSelector,
        "Set the breakpoint by ObjC selector name. Can be repeated multiple times to make one breakpoint for multiple Selectors." },

    { LLDB_OPT_SET_6, true, "method", 'M', required_argument, NULL, 0, eArgTypeMethod,
        "Set the breakpoint by C++ method names.  Can be repeated multiple times to make one breakpoint for multiple methods." },

    { LLDB_OPT_SET_7, true, "func-regex", 'r', required_argument, NULL, 0, eArgTypeRegularExpression,
        "Set the breakpoint by function name, evaluating a regular-expression to find the function name(s)." },

    { LLDB_OPT_SET_8, true, "basename", 'b', required_argument, NULL, CommandCompletions::eSymbolCompletion, eArgTypeFunctionName,
        "Set the breakpoint by function basename (C++ namespaces and arguments will be ignored). "
        "Can be repeated multiple times to make one breakpoint for multiple symbols." },

    { LLDB_OPT_SET_9, true, "source-pattern-regexp", 'p', required_argument, NULL, 0, eArgTypeRegularExpression,
        "Set the breakpoint specifying a regular expression to match a pattern in the source text in a given source file." },

    { LLDB_OPT_SET_10, true, "language-exception", 'E', required_argument, NULL, 0, eArgTypeLanguage,
        "Set the breakpoint on exceptions thrown by the specified language (without options, on throw but not catch.)" },

    { LLDB_OPT_SET_10, false, "on-throw", 'w', required_argument, NULL, 0, eArgTypeBoolean,
        "Set the breakpoint on exception throW." },

    { LLDB_OPT_SET_10, false, "on-catch", 'h', required_argument, NULL, 0, eArgTypeBoolean,
        "Set the breakpoint on exception catcH." },

    { LLDB_OPT_SKIP_PROLOGUE, false, "skip-prologue", 'K', required_argument, NULL, 0, eArgTypeBoolean,
        "sKip the prologue if the breakpoint is at the beginning of a function.  If not set the target.skip-prologue setting is used." },

    { 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};

//-------------------------------------------------------------------------
// CommandObjectBreakpointModify
//-------------------------------------------------------------------------
#pragma mark Modify

class CommandObjectBreakpointModify : public CommandObjectParsed
{
public:

    CommandObjectBreakpointModify (CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter,
                             "breakpoint modify", 
                             "Modify the options on a breakpoint or set of breakpoints in the executable.  "
                             "If no breakpoint is specified, acts on the last created breakpoint.  "
                             "With the exception of -e, -d and -i, passing an empty argument clears the modification.", 
                             NULL),
        m_options (interpreter)
    {
        CommandArgumentEntry arg;
        CommandObject::AddIDsArgumentData(arg, eArgTypeBreakpointID, eArgTypeBreakpointIDRange);
        // Add the entry for the first argument for this command to the object's arguments vector.
        m_arguments.push_back (arg);   
    }


    virtual
    ~CommandObjectBreakpointModify () {}

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
            m_ignore_count (0),
            m_thread_id(LLDB_INVALID_THREAD_ID),
            m_thread_id_passed(false),
            m_thread_index (UINT32_MAX),
            m_thread_index_passed(false),
            m_thread_name(),
            m_queue_name(),
            m_condition (),
            m_one_shot (false),
            m_enable_passed (false),
            m_enable_value (false),
            m_name_passed (false),
            m_queue_passed (false),
            m_condition_passed (false),
            m_one_shot_passed (false)
        {
        }

        virtual
        ~CommandOptions () {}

        virtual Error
        SetOptionValue (uint32_t option_idx, const char *option_arg)
        {
            Error error;
            char short_option = (char) m_getopt_table[option_idx].val;

            switch (short_option)
            {
                case 'c':
                    if (option_arg != NULL)
                        m_condition.assign (option_arg);
                    else
                        m_condition.clear();
                    m_condition_passed = true;
                    break;
                case 'd':
                    m_enable_passed = true;
                    m_enable_value = false;
                    break;
                case 'e':
                    m_enable_passed = true;
                    m_enable_value = true;
                    break;
                case 'i':
                {
                    m_ignore_count = Args::StringToUInt32(option_arg, UINT32_MAX, 0);
                    if (m_ignore_count == UINT32_MAX)
                       error.SetErrorStringWithFormat ("invalid ignore count '%s'", option_arg);
                }
                break;
                case 'o':
                {
                    bool value, success;
                    value = Args::StringToBoolean(option_arg, false, &success);
                    if (success)
                    {
                        m_one_shot_passed = true;
                        m_one_shot = value;
                    }
                    else
                        error.SetErrorStringWithFormat("invalid boolean value '%s' passed for -o option", option_arg);
                }
                break;
                case 't' :
                {
                    if (option_arg[0] == '\0')
                    {
                        m_thread_id = LLDB_INVALID_THREAD_ID;
                        m_thread_id_passed = true;
                    }
                    else
                    {
                        m_thread_id = Args::StringToUInt64(option_arg, LLDB_INVALID_THREAD_ID, 0);
                        if (m_thread_id == LLDB_INVALID_THREAD_ID)
                           error.SetErrorStringWithFormat ("invalid thread id string '%s'", option_arg);
                        else
                            m_thread_id_passed = true;
                    }
                }
                break;
                case 'T':
                    if (option_arg != NULL)
                        m_thread_name.assign (option_arg);
                    else
                        m_thread_name.clear();
                    m_name_passed = true;
                    break;
                case 'q':
                    if (option_arg != NULL)
                        m_queue_name.assign (option_arg);
                    else
                        m_queue_name.clear();
                    m_queue_passed = true;
                    break;
                case 'x':
                {
                    if (option_arg[0] == '\n')
                    {
                        m_thread_index = UINT32_MAX;
                        m_thread_index_passed = true;
                    }
                    else
                    {
                        m_thread_index = Args::StringToUInt32 (option_arg, UINT32_MAX, 0);
                        if (m_thread_id == UINT32_MAX)
                           error.SetErrorStringWithFormat ("invalid thread index string '%s'", option_arg);
                        else
                            m_thread_index_passed = true;
                    }
                }
                break;
                default:
                    error.SetErrorStringWithFormat ("unrecognized option '%c'", short_option);
                    break;
            }

            return error;
        }
        void
        OptionParsingStarting ()
        {
            m_ignore_count = 0;
            m_thread_id = LLDB_INVALID_THREAD_ID;
            m_thread_id_passed = false;
            m_thread_index = UINT32_MAX;
            m_thread_index_passed = false;
            m_thread_name.clear();
            m_queue_name.clear();
            m_condition.clear();
            m_one_shot = false;
            m_enable_passed = false;
            m_queue_passed = false;
            m_name_passed = false;
            m_condition_passed = false;
            m_one_shot_passed = false;
        }
        
        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }
        

        // Options table: Required for subclasses of Options.

        static OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.

        uint32_t m_ignore_count;
        lldb::tid_t m_thread_id;
        bool m_thread_id_passed;
        uint32_t m_thread_index;
        bool m_thread_index_passed;
        std::string m_thread_name;
        std::string m_queue_name;
        std::string m_condition;
        bool m_one_shot;
        bool m_enable_passed;
        bool m_enable_value;
        bool m_name_passed;
        bool m_queue_passed;
        bool m_condition_passed;
        bool m_one_shot_passed;

    };

protected:
    virtual bool
    DoExecute (Args& command, CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target == NULL)
        {
            result.AppendError ("Invalid target.  No existing target or breakpoints.");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        Mutex::Locker locker;
        target->GetBreakpointList().GetListMutex(locker);
        
        BreakpointIDList valid_bp_ids;

        CommandObjectMultiwordBreakpoint::VerifyBreakpointIDs (command, target, result, &valid_bp_ids);

        if (result.Succeeded())
        {
            const size_t count = valid_bp_ids.GetSize(); 
            for (size_t i = 0; i < count; ++i)
            {
                BreakpointID cur_bp_id = valid_bp_ids.GetBreakpointIDAtIndex (i);

                if (cur_bp_id.GetBreakpointID() != LLDB_INVALID_BREAK_ID)
                {
                    Breakpoint *bp = target->GetBreakpointByID (cur_bp_id.GetBreakpointID()).get();
                    if (cur_bp_id.GetLocationID() != LLDB_INVALID_BREAK_ID)
                    {
                        BreakpointLocation *location = bp->FindLocationByID (cur_bp_id.GetLocationID()).get();
                        if (location)
                        {
                            if (m_options.m_thread_id_passed)
                                location->SetThreadID (m_options.m_thread_id);
                                
                            if (m_options.m_thread_index_passed)
                                location->SetThreadIndex(m_options.m_thread_index);
                            
                            if (m_options.m_name_passed)
                                location->SetThreadName(m_options.m_thread_name.c_str());
                            
                            if (m_options.m_queue_passed)
                                location->SetQueueName(m_options.m_queue_name.c_str());
                                
                            if (m_options.m_ignore_count != 0)
                                location->SetIgnoreCount(m_options.m_ignore_count);
                                
                            if (m_options.m_enable_passed)
                                location->SetEnabled (m_options.m_enable_value);
                                
                            if (m_options.m_condition_passed)
                                location->SetCondition (m_options.m_condition.c_str());
                        }
                    }
                    else
                    {
                        if (m_options.m_thread_id_passed)
                            bp->SetThreadID (m_options.m_thread_id);
                            
                        if (m_options.m_thread_index_passed)
                            bp->SetThreadIndex(m_options.m_thread_index);
                        
                        if (m_options.m_name_passed)
                            bp->SetThreadName(m_options.m_thread_name.c_str());
                        
                        if (m_options.m_queue_passed)
                            bp->SetQueueName(m_options.m_queue_name.c_str());
                            
                        if (m_options.m_ignore_count != 0)
                            bp->SetIgnoreCount(m_options.m_ignore_count);
                            
                        if (m_options.m_enable_passed)
                            bp->SetEnabled (m_options.m_enable_value);
                            
                        if (m_options.m_condition_passed)
                            bp->SetCondition (m_options.m_condition.c_str());
                    }
                }
            }
        }
        
        return result.Succeeded();
    }

private:
    CommandOptions m_options;
};

#pragma mark Modify::CommandOptions
OptionDefinition
CommandObjectBreakpointModify::CommandOptions::g_option_table[] =
{
{ LLDB_OPT_SET_ALL, false, "ignore-count", 'i', required_argument, NULL, 0, eArgTypeCount, "Set the number of times this breakpoint is skipped before stopping." },
{ LLDB_OPT_SET_ALL, false, "one-shot",     'o', required_argument, NULL, 0, eArgTypeBoolean, "The breakpoint is deleted the first time it stop causes a stop." },
{ LLDB_OPT_SET_ALL, false, "thread-index", 'x', required_argument, NULL, 0, eArgTypeThreadIndex, "The breakpoint stops only for the thread whose indeX matches this argument."},
{ LLDB_OPT_SET_ALL, false, "thread-id",    't', required_argument, NULL, 0, eArgTypeThreadID, "The breakpoint stops only for the thread whose TID matches this argument."},
{ LLDB_OPT_SET_ALL, false, "thread-name",  'T', required_argument, NULL, 0, eArgTypeThreadName, "The breakpoint stops only for the thread whose thread name matches this argument."},
{ LLDB_OPT_SET_ALL, false, "queue-name",   'q', required_argument, NULL, 0, eArgTypeQueueName, "The breakpoint stops only for threads in the queue whose name is given by this argument."},
{ LLDB_OPT_SET_ALL, false, "condition",    'c', required_argument, NULL, 0, eArgTypeExpression, "The breakpoint stops only if this condition expression evaluates to true."},
{ LLDB_OPT_SET_1,   false, "enable",       'e', no_argument,       NULL, 0, eArgTypeNone, "Enable the breakpoint."},
{ LLDB_OPT_SET_2,   false, "disable",      'd', no_argument,       NULL, 0, eArgTypeNone, "Disable the breakpoint."},
{ 0,                false, NULL,            0 , 0,                 NULL, 0, eArgTypeNone, NULL }
};

//-------------------------------------------------------------------------
// CommandObjectBreakpointEnable
//-------------------------------------------------------------------------
#pragma mark Enable

class CommandObjectBreakpointEnable : public CommandObjectParsed
{
public:
    CommandObjectBreakpointEnable (CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter,
                             "enable",
                             "Enable the specified disabled breakpoint(s). If no breakpoints are specified, enable all of them.",
                             NULL)
    {
        CommandArgumentEntry arg;
        CommandObject::AddIDsArgumentData(arg, eArgTypeBreakpointID, eArgTypeBreakpointIDRange);
        // Add the entry for the first argument for this command to the object's arguments vector.
        m_arguments.push_back (arg);   
    }


    virtual
    ~CommandObjectBreakpointEnable () {}

protected:
    virtual bool
    DoExecute (Args& command, CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target == NULL)
        {
            result.AppendError ("Invalid target.  No existing target or breakpoints.");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        Mutex::Locker locker;
        target->GetBreakpointList().GetListMutex(locker);

        const BreakpointList &breakpoints = target->GetBreakpointList();

        size_t num_breakpoints = breakpoints.GetSize();

        if (num_breakpoints == 0)
        {
            result.AppendError ("No breakpoints exist to be enabled.");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        if (command.GetArgumentCount() == 0)
        {
            // No breakpoint selected; enable all currently set breakpoints.
            target->EnableAllBreakpoints ();
            result.AppendMessageWithFormat ("All breakpoints enabled. (%lu breakpoints)\n", num_breakpoints);
            result.SetStatus (eReturnStatusSuccessFinishNoResult);
        }
        else
        {
            // Particular breakpoint selected; enable that breakpoint.
            BreakpointIDList valid_bp_ids;
            CommandObjectMultiwordBreakpoint::VerifyBreakpointIDs (command, target, result, &valid_bp_ids);

            if (result.Succeeded())
            {
                int enable_count = 0;
                int loc_count = 0;
                const size_t count = valid_bp_ids.GetSize();
                for (size_t i = 0; i < count; ++i)
                {
                    BreakpointID cur_bp_id = valid_bp_ids.GetBreakpointIDAtIndex (i);

                    if (cur_bp_id.GetBreakpointID() != LLDB_INVALID_BREAK_ID)
                    {
                        Breakpoint *breakpoint = target->GetBreakpointByID (cur_bp_id.GetBreakpointID()).get();
                        if (cur_bp_id.GetLocationID() != LLDB_INVALID_BREAK_ID)
                        {
                            BreakpointLocation *location = breakpoint->FindLocationByID (cur_bp_id.GetLocationID()).get();
                            if (location)
                            {
                                location->SetEnabled (true);
                                ++loc_count;
                            }
                        }
                        else
                        {
                            breakpoint->SetEnabled (true);
                            ++enable_count;
                        }
                    }
                }
                result.AppendMessageWithFormat ("%d breakpoints enabled.\n", enable_count + loc_count);
                result.SetStatus (eReturnStatusSuccessFinishNoResult);
            }
        }

        return result.Succeeded();
    }
};

//-------------------------------------------------------------------------
// CommandObjectBreakpointDisable
//-------------------------------------------------------------------------
#pragma mark Disable

class CommandObjectBreakpointDisable : public CommandObjectParsed
{
public:
    CommandObjectBreakpointDisable (CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter,
                             "breakpoint disable",
                             "Disable the specified breakpoint(s) without removing it/them.  If no breakpoints are specified, disable them all.",
                             NULL)
    {
        CommandArgumentEntry arg;
        CommandObject::AddIDsArgumentData(arg, eArgTypeBreakpointID, eArgTypeBreakpointIDRange);
        // Add the entry for the first argument for this command to the object's arguments vector.
        m_arguments.push_back (arg);   
    }


    virtual
    ~CommandObjectBreakpointDisable () {}

protected:
    virtual bool
    DoExecute (Args& command, CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target == NULL)
        {
            result.AppendError ("Invalid target.  No existing target or breakpoints.");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        Mutex::Locker locker;
        target->GetBreakpointList().GetListMutex(locker);

        const BreakpointList &breakpoints = target->GetBreakpointList();
        size_t num_breakpoints = breakpoints.GetSize();

        if (num_breakpoints == 0)
        {
            result.AppendError ("No breakpoints exist to be disabled.");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        if (command.GetArgumentCount() == 0)
        {
            // No breakpoint selected; disable all currently set breakpoints.
            target->DisableAllBreakpoints ();
            result.AppendMessageWithFormat ("All breakpoints disabled. (%lu breakpoints)\n", num_breakpoints);
            result.SetStatus (eReturnStatusSuccessFinishNoResult);
        }
        else
        {
            // Particular breakpoint selected; disable that breakpoint.
            BreakpointIDList valid_bp_ids;

            CommandObjectMultiwordBreakpoint::VerifyBreakpointIDs (command, target, result, &valid_bp_ids);

            if (result.Succeeded())
            {
                int disable_count = 0;
                int loc_count = 0;
                const size_t count = valid_bp_ids.GetSize();
                for (size_t i = 0; i < count; ++i)
                {
                    BreakpointID cur_bp_id = valid_bp_ids.GetBreakpointIDAtIndex (i);

                    if (cur_bp_id.GetBreakpointID() != LLDB_INVALID_BREAK_ID)
                    {
                        Breakpoint *breakpoint = target->GetBreakpointByID (cur_bp_id.GetBreakpointID()).get();
                        if (cur_bp_id.GetLocationID() != LLDB_INVALID_BREAK_ID)
                        {
                            BreakpointLocation *location = breakpoint->FindLocationByID (cur_bp_id.GetLocationID()).get();
                            if (location)
                            {
                                location->SetEnabled (false);
                                ++loc_count;
                            }
                        }
                        else
                        {
                            breakpoint->SetEnabled (false);
                            ++disable_count;
                        }
                    }
                }
                result.AppendMessageWithFormat ("%d breakpoints disabled.\n", disable_count + loc_count);
                result.SetStatus (eReturnStatusSuccessFinishNoResult);
            }
        }

        return result.Succeeded();
    }

};

//-------------------------------------------------------------------------
// CommandObjectBreakpointList
//-------------------------------------------------------------------------
#pragma mark List

class CommandObjectBreakpointList : public CommandObjectParsed
{
public:
    CommandObjectBreakpointList (CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter, 
                             "breakpoint list",
                             "List some or all breakpoints at configurable levels of detail.",
                             NULL),
        m_options (interpreter)
    {
        CommandArgumentEntry arg;
        CommandArgumentData bp_id_arg;

        // Define the first (and only) variant of this arg.
        bp_id_arg.arg_type = eArgTypeBreakpointID;
        bp_id_arg.arg_repetition = eArgRepeatOptional;

        // There is only one variant this argument could be; put it into the argument entry.
        arg.push_back (bp_id_arg);

        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg);
    }


    virtual
    ~CommandObjectBreakpointList () {}

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
            m_level (lldb::eDescriptionLevelBrief)  // Breakpoint List defaults to brief descriptions
        {
        }

        virtual
        ~CommandOptions () {}

        virtual Error
        SetOptionValue (uint32_t option_idx, const char *option_arg)
        {
            Error error;
            char short_option = (char) m_getopt_table[option_idx].val;

            switch (short_option)
            {
                case 'b':
                    m_level = lldb::eDescriptionLevelBrief;
                    break;
                case 'f':
                    m_level = lldb::eDescriptionLevelFull;
                    break;
                case 'v':
                    m_level = lldb::eDescriptionLevelVerbose;
                    break;
                case 'i':
                    m_internal = true;
                    break;
                default:
                    error.SetErrorStringWithFormat ("unrecognized option '%c'", short_option);
                    break;
            }

            return error;
        }

        void
        OptionParsingStarting ()
        {
            m_level = lldb::eDescriptionLevelFull;
            m_internal = false;
        }

        const OptionDefinition *
        GetDefinitions ()
        {
            return g_option_table;
        }

        // Options table: Required for subclasses of Options.

        static OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.

        lldb::DescriptionLevel m_level;

        bool m_internal;
    };

protected:
    virtual bool
    DoExecute (Args& command, CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target == NULL)
        {
            result.AppendError ("Invalid target. No current target or breakpoints.");
            result.SetStatus (eReturnStatusSuccessFinishNoResult);
            return true;
        }

        const BreakpointList &breakpoints = target->GetBreakpointList(m_options.m_internal);
        Mutex::Locker locker;
        target->GetBreakpointList(m_options.m_internal).GetListMutex(locker);

        size_t num_breakpoints = breakpoints.GetSize();

        if (num_breakpoints == 0)
        {
            result.AppendMessage ("No breakpoints currently set.");
            result.SetStatus (eReturnStatusSuccessFinishNoResult);
            return true;
        }

        Stream &output_stream = result.GetOutputStream();

        if (command.GetArgumentCount() == 0)
        {
            // No breakpoint selected; show info about all currently set breakpoints.
            result.AppendMessage ("Current breakpoints:");
            for (size_t i = 0; i < num_breakpoints; ++i)
            {
                Breakpoint *breakpoint = breakpoints.GetBreakpointAtIndex (i).get();
                AddBreakpointDescription (&output_stream, breakpoint, m_options.m_level);
            }
            result.SetStatus (eReturnStatusSuccessFinishNoResult);
        }
        else
        {
            // Particular breakpoints selected; show info about that breakpoint.
            BreakpointIDList valid_bp_ids;
            CommandObjectMultiwordBreakpoint::VerifyBreakpointIDs (command, target, result, &valid_bp_ids);

            if (result.Succeeded())
            {
                for (size_t i = 0; i < valid_bp_ids.GetSize(); ++i)
                {
                    BreakpointID cur_bp_id = valid_bp_ids.GetBreakpointIDAtIndex (i);
                    Breakpoint *breakpoint = target->GetBreakpointByID (cur_bp_id.GetBreakpointID()).get();
                    AddBreakpointDescription (&output_stream, breakpoint, m_options.m_level);
                }
                result.SetStatus (eReturnStatusSuccessFinishNoResult);
            }
            else
            {
                result.AppendError ("Invalid breakpoint id.");
                result.SetStatus (eReturnStatusFailed);
            }
        }

        return result.Succeeded();
    }

private:
    CommandOptions m_options;
};

#pragma mark List::CommandOptions
OptionDefinition
CommandObjectBreakpointList::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_ALL, false, "internal", 'i', no_argument, NULL, 0, eArgTypeNone,
        "Show debugger internal breakpoints" },

    { LLDB_OPT_SET_1, false, "brief",    'b', no_argument, NULL, 0, eArgTypeNone,
        "Give a brief description of the breakpoint (no location info)."},

    // FIXME: We need to add an "internal" command, and then add this sort of thing to it.
    // But I need to see it for now, and don't want to wait.
    { LLDB_OPT_SET_2, false, "full",    'f', no_argument, NULL, 0, eArgTypeNone,
        "Give a full description of the breakpoint and its locations."},

    { LLDB_OPT_SET_3, false, "verbose", 'v', no_argument, NULL, 0, eArgTypeNone,
        "Explain everything we know about the breakpoint (for debugging debugger bugs)." },

    { 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};

//-------------------------------------------------------------------------
// CommandObjectBreakpointClear
//-------------------------------------------------------------------------
#pragma mark Clear

class CommandObjectBreakpointClear : public CommandObjectParsed
{
public:

    typedef enum BreakpointClearType
    {
        eClearTypeInvalid,
        eClearTypeFileAndLine
    } BreakpointClearType;

    CommandObjectBreakpointClear (CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter,
                             "breakpoint clear", 
                             "Clears a breakpoint or set of breakpoints in the executable.", 
                             "breakpoint clear <cmd-options>"),
        m_options (interpreter)
    {
    }

    virtual
    ~CommandObjectBreakpointClear () {}

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
            m_filename (),
            m_line_num (0)
        {
        }

        virtual
        ~CommandOptions () {}

        virtual Error
        SetOptionValue (uint32_t option_idx, const char *option_arg)
        {
            Error error;
            char short_option = (char) m_getopt_table[option_idx].val;

            switch (short_option)
            {
                case 'f':
                    m_filename.assign (option_arg);
                    break;

                case 'l':
                    m_line_num = Args::StringToUInt32 (option_arg, 0);
                    break;

                default:
                    error.SetErrorStringWithFormat ("unrecognized option '%c'", short_option);
                    break;
            }

            return error;
        }

        void
        OptionParsingStarting ()
        {
            m_filename.clear();
            m_line_num = 0;
        }

        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }

        // Options table: Required for subclasses of Options.

        static OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.

        std::string m_filename;
        uint32_t m_line_num;

    };

protected:
    virtual bool
    DoExecute (Args& command, CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target == NULL)
        {
            result.AppendError ("Invalid target. No existing target or breakpoints.");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        // The following are the various types of breakpoints that could be cleared:
        //   1). -f -l (clearing breakpoint by source location)

        BreakpointClearType break_type = eClearTypeInvalid;

        if (m_options.m_line_num != 0)
            break_type = eClearTypeFileAndLine;

        Mutex::Locker locker;
        target->GetBreakpointList().GetListMutex(locker);

        BreakpointList &breakpoints = target->GetBreakpointList();
        size_t num_breakpoints = breakpoints.GetSize();

        // Early return if there's no breakpoint at all.
        if (num_breakpoints == 0)
        {
            result.AppendError ("Breakpoint clear: No breakpoint cleared.");
            result.SetStatus (eReturnStatusFailed);
            return result.Succeeded();
        }

        // Find matching breakpoints and delete them.

        // First create a copy of all the IDs.
        std::vector<break_id_t> BreakIDs;
        for (size_t i = 0; i < num_breakpoints; ++i)
            BreakIDs.push_back(breakpoints.GetBreakpointAtIndex(i).get()->GetID());

        int num_cleared = 0;
        StreamString ss;
        switch (break_type)
        {
            case eClearTypeFileAndLine: // Breakpoint by source position
                {
                    const ConstString filename(m_options.m_filename.c_str());
                    BreakpointLocationCollection loc_coll;

                    for (size_t i = 0; i < num_breakpoints; ++i)
                    {
                        Breakpoint *bp = breakpoints.FindBreakpointByID(BreakIDs[i]).get();
                        
                        if (bp->GetMatchingFileLine(filename, m_options.m_line_num, loc_coll))
                        {
                            // If the collection size is 0, it's a full match and we can just remove the breakpoint.
                            if (loc_coll.GetSize() == 0)
                            {
                                bp->GetDescription(&ss, lldb::eDescriptionLevelBrief);
                                ss.EOL();
                                target->RemoveBreakpointByID (bp->GetID());
                                ++num_cleared;
                            }
                        }
                    }
                }
                break;

            default:
                break;
        }

        if (num_cleared > 0)
        {
            Stream &output_stream = result.GetOutputStream();
            output_stream.Printf ("%d breakpoints cleared:\n", num_cleared);
            output_stream << ss.GetData();
            output_stream.EOL();
            result.SetStatus (eReturnStatusSuccessFinishNoResult);
        }
        else
        {
            result.AppendError ("Breakpoint clear: No breakpoint cleared.");
            result.SetStatus (eReturnStatusFailed);
        }

        return result.Succeeded();
    }

private:
    CommandOptions m_options;
};

#pragma mark Clear::CommandOptions

OptionDefinition
CommandObjectBreakpointClear::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_1, false, "file", 'f', required_argument, NULL, CommandCompletions::eSourceFileCompletion, eArgTypeFilename,
        "Specify the breakpoint by source location in this particular file."},

    { LLDB_OPT_SET_1, true, "line", 'l', required_argument, NULL, 0, eArgTypeLineNum,
        "Specify the breakpoint by source location at this particular line."},

    { 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};

//-------------------------------------------------------------------------
// CommandObjectBreakpointDelete
//-------------------------------------------------------------------------
#pragma mark Delete

class CommandObjectBreakpointDelete : public CommandObjectParsed
{
public:
    CommandObjectBreakpointDelete (CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter,
                             "breakpoint delete",
                             "Delete the specified breakpoint(s).  If no breakpoints are specified, delete them all.",
                             NULL)
    {
        CommandArgumentEntry arg;
        CommandObject::AddIDsArgumentData(arg, eArgTypeBreakpointID, eArgTypeBreakpointIDRange);
        // Add the entry for the first argument for this command to the object's arguments vector.
        m_arguments.push_back (arg);   
    }

    virtual
    ~CommandObjectBreakpointDelete () {}

protected:
    virtual bool
    DoExecute (Args& command, CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target == NULL)
        {
            result.AppendError ("Invalid target. No existing target or breakpoints.");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        Mutex::Locker locker;
        target->GetBreakpointList().GetListMutex(locker);
        
        const BreakpointList &breakpoints = target->GetBreakpointList();

        size_t num_breakpoints = breakpoints.GetSize();

        if (num_breakpoints == 0)
        {
            result.AppendError ("No breakpoints exist to be deleted.");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        if (command.GetArgumentCount() == 0)
        {
            if (!m_interpreter.Confirm ("About to delete all breakpoints, do you want to do that?", true))
            {
                result.AppendMessage("Operation cancelled...");
            }
            else
            {
                target->RemoveAllBreakpoints ();
                result.AppendMessageWithFormat ("All breakpoints removed. (%lu breakpoints)\n", num_breakpoints);
            }
            result.SetStatus (eReturnStatusSuccessFinishNoResult);
        }
        else
        {
            // Particular breakpoint selected; disable that breakpoint.
            BreakpointIDList valid_bp_ids;
            CommandObjectMultiwordBreakpoint::VerifyBreakpointIDs (command, target, result, &valid_bp_ids);

            if (result.Succeeded())
            {
                int delete_count = 0;
                int disable_count = 0;
                const size_t count = valid_bp_ids.GetSize();
                for (size_t i = 0; i < count; ++i)
                {
                    BreakpointID cur_bp_id = valid_bp_ids.GetBreakpointIDAtIndex (i);

                    if (cur_bp_id.GetBreakpointID() != LLDB_INVALID_BREAK_ID)
                    {
                        if (cur_bp_id.GetLocationID() != LLDB_INVALID_BREAK_ID)
                        {
                            Breakpoint *breakpoint = target->GetBreakpointByID (cur_bp_id.GetBreakpointID()).get();
                            BreakpointLocation *location = breakpoint->FindLocationByID (cur_bp_id.GetLocationID()).get();
                            // It makes no sense to try to delete individual locations, so we disable them instead.
                            if (location)
                            {
                                location->SetEnabled (false);
                                ++disable_count;
                            }
                        }
                        else
                        {
                            target->RemoveBreakpointByID (cur_bp_id.GetBreakpointID());
                            ++delete_count;
                        }
                    }
                }
                result.AppendMessageWithFormat ("%d breakpoints deleted; %d breakpoint locations disabled.\n",
                                               delete_count, disable_count);
                result.SetStatus (eReturnStatusSuccessFinishNoResult);
            }
        }
        return result.Succeeded();
    }
};

//-------------------------------------------------------------------------
// CommandObjectMultiwordBreakpoint
//-------------------------------------------------------------------------
#pragma mark MultiwordBreakpoint

CommandObjectMultiwordBreakpoint::CommandObjectMultiwordBreakpoint (CommandInterpreter &interpreter) :
    CommandObjectMultiword (interpreter, 
                            "breakpoint",
                            "A set of commands for operating on breakpoints. Also see _regexp-break.",
                            "breakpoint <command> [<command-options>]")
{
    CommandObjectSP list_command_object (new CommandObjectBreakpointList (interpreter));
    CommandObjectSP enable_command_object (new CommandObjectBreakpointEnable (interpreter));
    CommandObjectSP disable_command_object (new CommandObjectBreakpointDisable (interpreter));
    CommandObjectSP clear_command_object (new CommandObjectBreakpointClear (interpreter));
    CommandObjectSP delete_command_object (new CommandObjectBreakpointDelete (interpreter));
    CommandObjectSP set_command_object (new CommandObjectBreakpointSet (interpreter));
    CommandObjectSP command_command_object (new CommandObjectBreakpointCommand (interpreter));
    CommandObjectSP modify_command_object (new CommandObjectBreakpointModify(interpreter));

    list_command_object->SetCommandName ("breakpoint list");
    enable_command_object->SetCommandName("breakpoint enable");
    disable_command_object->SetCommandName("breakpoint disable");
    clear_command_object->SetCommandName("breakpoint clear");
    delete_command_object->SetCommandName("breakpoint delete");
    set_command_object->SetCommandName("breakpoint set");
    command_command_object->SetCommandName ("breakpoint command");
    modify_command_object->SetCommandName ("breakpoint modify");

    LoadSubCommand ("list",       list_command_object);
    LoadSubCommand ("enable",     enable_command_object);
    LoadSubCommand ("disable",    disable_command_object);
    LoadSubCommand ("clear",      clear_command_object);
    LoadSubCommand ("delete",     delete_command_object);
    LoadSubCommand ("set",        set_command_object);
    LoadSubCommand ("command",    command_command_object);
    LoadSubCommand ("modify",     modify_command_object);
}

CommandObjectMultiwordBreakpoint::~CommandObjectMultiwordBreakpoint ()
{
}

void
CommandObjectMultiwordBreakpoint::VerifyBreakpointIDs (Args &args, Target *target, CommandReturnObject &result,
                                                         BreakpointIDList *valid_ids)
{
    // args can be strings representing 1). integers (for breakpoint ids)
    //                                  2). the full breakpoint & location canonical representation
    //                                  3). the word "to" or a hyphen, representing a range (in which case there
    //                                      had *better* be an entry both before & after of one of the first two types.
    // If args is empty, we will use the last created breakpoint (if there is one.)

    Args temp_args;

    if (args.GetArgumentCount() == 0)
    {
        if (target->GetLastCreatedBreakpoint())
        {
            valid_ids->AddBreakpointID (BreakpointID(target->GetLastCreatedBreakpoint()->GetID(), LLDB_INVALID_BREAK_ID));
            result.SetStatus (eReturnStatusSuccessFinishNoResult);
        } 
        else
        {   
            result.AppendError("No breakpoint specified and no last created breakpoint.");
            result.SetStatus (eReturnStatusFailed);
        }
        return;
    }
    
    // Create a new Args variable to use; copy any non-breakpoint-id-ranges stuff directly from the old ARGS to
    // the new TEMP_ARGS.  Do not copy breakpoint id range strings over; instead generate a list of strings for
    // all the breakpoint ids in the range, and shove all of those breakpoint id strings into TEMP_ARGS.

    BreakpointIDList::FindAndReplaceIDRanges (args, target, result, temp_args);

    // NOW, convert the list of breakpoint id strings in TEMP_ARGS into an actual BreakpointIDList:

    valid_ids->InsertStringArray (temp_args.GetConstArgumentVector(), temp_args.GetArgumentCount(), result);

    // At this point,  all of the breakpoint ids that the user passed in have been converted to breakpoint IDs
    // and put into valid_ids.

    if (result.Succeeded())
    {
        // Now that we've converted everything from args into a list of breakpoint ids, go through our tentative list
        // of breakpoint id's and verify that they correspond to valid/currently set breakpoints.

        const size_t count = valid_ids->GetSize();
        for (size_t i = 0; i < count; ++i)
        {
            BreakpointID cur_bp_id = valid_ids->GetBreakpointIDAtIndex (i);
            Breakpoint *breakpoint = target->GetBreakpointByID (cur_bp_id.GetBreakpointID()).get();
            if (breakpoint != NULL)
            {
                int num_locations = breakpoint->GetNumLocations();
                if (cur_bp_id.GetLocationID() > num_locations)
                {
                    StreamString id_str;
                    BreakpointID::GetCanonicalReference (&id_str, 
                                                         cur_bp_id.GetBreakpointID(),
                                                         cur_bp_id.GetLocationID());
                    i = valid_ids->GetSize() + 1;
                    result.AppendErrorWithFormat ("'%s' is not a currently valid breakpoint/location id.\n",
                                                 id_str.GetData());
                    result.SetStatus (eReturnStatusFailed);
                }
            }
            else
            {
                i = valid_ids->GetSize() + 1;
                result.AppendErrorWithFormat ("'%d' is not a currently valid breakpoint id.\n", cur_bp_id.GetBreakpointID());
                result.SetStatus (eReturnStatusFailed);
            }
        }
    }
}
