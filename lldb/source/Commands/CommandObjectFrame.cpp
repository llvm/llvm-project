//===-- CommandObjectFrame.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectFrame.h"

// C Includes
// C++ Includes
#include <string>
// Other libraries and framework includes
// Project includes
#include "lldb/Core/DataVisualization.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/Timer.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Core/ValueObjectVariable.h"
#include "lldb/Host/Host.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Interpreter/OptionGroupFormat.h"
#include "lldb/Interpreter/OptionGroupValueObjectDisplay.h"
#include "lldb/Interpreter/OptionGroupVariable.h"
#include "lldb/Symbol/ClangASTType.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/Variable.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

#pragma mark CommandObjectFrameInfo

//-------------------------------------------------------------------------
// CommandObjectFrameInfo
//-------------------------------------------------------------------------

class CommandObjectFrameInfo : public CommandObjectParsed
{
public:

    CommandObjectFrameInfo (CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter,
                             "frame info",
                             "List information about the currently selected frame in the current thread.",
                             "frame info",
                             eFlagProcessMustBeLaunched | eFlagProcessMustBePaused)
    {
    }

    ~CommandObjectFrameInfo ()
    {
    }

protected:
    bool
    DoExecute (Args& command,
             CommandReturnObject &result)
    {
        ExecutionContext exe_ctx(m_interpreter.GetExecutionContext());
        StackFrame *frame = exe_ctx.GetFramePtr();
        if (frame)
        {
            frame->DumpUsingSettingsFormat (&result.GetOutputStream());
            result.SetStatus (eReturnStatusSuccessFinishResult);
        }
        else
        {
            result.AppendError ("no current frame");
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }
};

#pragma mark CommandObjectFrameSelect

//-------------------------------------------------------------------------
// CommandObjectFrameSelect
//-------------------------------------------------------------------------

class CommandObjectFrameSelect : public CommandObjectParsed
{
public:

   class CommandOptions : public Options
    {
    public:

        CommandOptions (CommandInterpreter &interpreter) :
            Options(interpreter)
        {
            OptionParsingStarting ();
        }

        virtual
        ~CommandOptions ()
        {
        }

        virtual Error
        SetOptionValue (uint32_t option_idx, const char *option_arg)
        {
            Error error;
            bool success = false;
            char short_option = (char) m_getopt_table[option_idx].val;
            switch (short_option)
            {
            case 'r':   
                relative_frame_offset = Args::StringToSInt32 (option_arg, INT32_MIN, 0, &success);
                if (!success)
                    error.SetErrorStringWithFormat ("invalid frame offset argument '%s'", option_arg);
                break;

            default:
                error.SetErrorStringWithFormat ("invalid short option character '%c'", short_option);
                break;
            }

            return error;
        }

        void
        OptionParsingStarting ()
        {
            relative_frame_offset = INT32_MIN;
        }

        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }

        // Options table: Required for subclasses of Options.

        static OptionDefinition g_option_table[];
        int32_t relative_frame_offset;
    };
    
    CommandObjectFrameSelect (CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter,
                             "frame select",
                             "Select a frame by index from within the current thread and make it the current frame.",
                             NULL,
                             eFlagProcessMustBeLaunched | eFlagProcessMustBePaused),
        m_options (interpreter)
    {
        CommandArgumentEntry arg;
        CommandArgumentData index_arg;

        // Define the first (and only) variant of this arg.
        index_arg.arg_type = eArgTypeFrameIndex;
        index_arg.arg_repetition = eArgRepeatOptional;

        // There is only one variant this argument could be; put it into the argument entry.
        arg.push_back (index_arg);

        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg);
    }

    ~CommandObjectFrameSelect ()
    {
    }

    virtual
    Options *
    GetOptions ()
    {
        return &m_options;
    }


protected:
    bool
    DoExecute (Args& command,
             CommandReturnObject &result)
    {
        ExecutionContext exe_ctx (m_interpreter.GetExecutionContext());
        Thread *thread = exe_ctx.GetThreadPtr();
        if (thread)
        {
            uint32_t frame_idx = UINT32_MAX;
            if (m_options.relative_frame_offset != INT32_MIN)
            {
                // The one and only argument is a signed relative frame index
                frame_idx = thread->GetSelectedFrameIndex ();
                if (frame_idx == UINT32_MAX)
                    frame_idx = 0;
                
                if (m_options.relative_frame_offset < 0)
                {
                    if (frame_idx >= -m_options.relative_frame_offset)
                        frame_idx += m_options.relative_frame_offset;
                    else
                    {
                        if (frame_idx == 0)
                        {
                            //If you are already at the bottom of the stack, then just warn and don't reset the frame.
                            result.AppendError("Already at the bottom of the stack");
                            result.SetStatus(eReturnStatusFailed);
                            return false;
                        }
                        else
                            frame_idx = 0;
                    }
                }
                else if (m_options.relative_frame_offset > 0)
                {
                    // I don't want "up 20" where "20" takes you past the top of the stack to produce
                    // an error, but rather to just go to the top.  So I have to count the stack here...
                    const uint32_t num_frames = thread->GetStackFrameCount();
                    if (num_frames - frame_idx > m_options.relative_frame_offset)
                        frame_idx += m_options.relative_frame_offset;
                    else
                    {
                        if (frame_idx == num_frames - 1)
                        {
                            //If we are already at the top of the stack, just warn and don't reset the frame.
                            result.AppendError("Already at the top of the stack");
                            result.SetStatus(eReturnStatusFailed);
                            return false;
                        }
                        else
                            frame_idx = num_frames - 1;
                    }
                }
            }
            else 
            {
                if (command.GetArgumentCount() == 1)
                {
                    const char *frame_idx_cstr = command.GetArgumentAtIndex(0);
                    frame_idx = Args::StringToUInt32 (frame_idx_cstr, UINT32_MAX, 0);
                }
                else if (command.GetArgumentCount() == 0)
                {
                    frame_idx = thread->GetSelectedFrameIndex ();
                    if (frame_idx == UINT32_MAX)
                    {
                        frame_idx = 0;
                    }
                }
                else
                {
                    result.AppendError ("invalid arguments.\n");
                    m_options.GenerateOptionUsage (result.GetErrorStream(), this);
                }
            }

            const bool broadcast = true;
            bool success = thread->SetSelectedFrameByIndex (frame_idx, broadcast);
            if (success)
            {
                exe_ctx.SetFrameSP(thread->GetSelectedFrame ());
                StackFrame *frame = exe_ctx.GetFramePtr();
                if (frame)
                {
                    bool already_shown = false;
                    SymbolContext frame_sc(frame->GetSymbolContext(eSymbolContextLineEntry));
                    if (m_interpreter.GetDebugger().GetUseExternalEditor() && frame_sc.line_entry.file && frame_sc.line_entry.line != 0)
                    {
                        already_shown = Host::OpenFileInExternalEditor (frame_sc.line_entry.file, frame_sc.line_entry.line);
                    }

                    bool show_frame_info = true;
                    bool show_source = !already_shown;
                    if (frame->GetStatus (result.GetOutputStream(), show_frame_info, show_source))
                    {
                        result.SetStatus (eReturnStatusSuccessFinishResult);
                        return result.Succeeded();
                    }
                }
            }
            result.AppendErrorWithFormat ("Frame index (%u) out of range.\n", frame_idx);
        }
        else
        {
            result.AppendError ("no current thread");
        }
        result.SetStatus (eReturnStatusFailed);
        return false;
    }
protected:

    CommandOptions m_options;
};

OptionDefinition
CommandObjectFrameSelect::CommandOptions::g_option_table[] =
{
{ LLDB_OPT_SET_1, false, "relative", 'r', required_argument, NULL, 0, eArgTypeOffset, "A relative frame index offset from the current frame index."},
{ 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};

#pragma mark CommandObjectFrameVariable
//----------------------------------------------------------------------
// List images with associated information
//----------------------------------------------------------------------
class CommandObjectFrameVariable : public CommandObjectParsed
{
public:

    CommandObjectFrameVariable (CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter,
                             "frame variable",
                             "Show frame variables. All argument and local variables "
                             "that are in scope will be shown when no arguments are given. "
                             "If any arguments are specified, they can be names of "
                             "argument, local, file static and file global variables. "
                             "Children of aggregate variables can be specified such as "
                             "'var->child.x'.",
                             NULL,
                             eFlagProcessMustBeLaunched | eFlagProcessMustBePaused),
        m_option_group (interpreter),
        m_option_variable(true), // Include the frame specific options by passing "true"
        m_option_format (eFormatDefault),
        m_varobj_options()
    {
        CommandArgumentEntry arg;
        CommandArgumentData var_name_arg;
        
        // Define the first (and only) variant of this arg.
        var_name_arg.arg_type = eArgTypeVarName;
        var_name_arg.arg_repetition = eArgRepeatStar;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg.push_back (var_name_arg);
        
        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg);
        
        m_option_group.Append (&m_option_variable, LLDB_OPT_SET_ALL, LLDB_OPT_SET_1);
        m_option_group.Append (&m_option_format, OptionGroupFormat::OPTION_GROUP_FORMAT | OptionGroupFormat::OPTION_GROUP_GDB_FMT, LLDB_OPT_SET_1);
        m_option_group.Append (&m_varobj_options, LLDB_OPT_SET_ALL, LLDB_OPT_SET_1);
        m_option_group.Finalize();
    }

    virtual
    ~CommandObjectFrameVariable ()
    {
    }

    virtual
    Options *
    GetOptions ()
    {
        return &m_option_group;
    }

protected:
    virtual bool
    DoExecute (Args& command, CommandReturnObject &result)
    {
        ExecutionContext exe_ctx(m_interpreter.GetExecutionContext());
        StackFrame *frame = exe_ctx.GetFramePtr();
        if (frame == NULL)
        {
            result.AppendError ("you must be stopped in a valid stack frame to view frame variables.");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        Stream &s = result.GetOutputStream();

        bool get_file_globals = true;
        
        // Be careful about the stack frame, if any summary formatter runs code, it might clear the StackFrameList
        // for the thread.  So hold onto a shared pointer to the frame so it stays alive.
        
        VariableList *variable_list = frame->GetVariableList (get_file_globals);

        VariableSP var_sp;
        ValueObjectSP valobj_sp;

        const char *name_cstr = NULL;
        size_t idx;
        
        TypeSummaryImplSP summary_format_sp;
        if (!m_option_variable.summary.IsCurrentValueEmpty())
            DataVisualization::NamedSummaryFormats::GetSummaryFormat(ConstString(m_option_variable.summary.GetCurrentValue()), summary_format_sp);
        else if (!m_option_variable.summary_string.IsCurrentValueEmpty())
            summary_format_sp.reset(new StringSummaryFormat(TypeSummaryImpl::Flags(),m_option_variable.summary_string.GetCurrentValue()));
        
        ValueObject::DumpValueObjectOptions options;
        
        options.SetMaximumPointerDepth(m_varobj_options.ptr_depth)
            .SetMaximumDepth(m_varobj_options.max_depth)
            .SetShowTypes(m_varobj_options.show_types)
            .SetShowLocation(m_varobj_options.show_location)
            .SetUseObjectiveC(m_varobj_options.use_objc)
            .SetUseDynamicType(m_varobj_options.use_dynamic)
            .SetUseSyntheticValue(m_varobj_options.use_synth)
            .SetFlatOutput(m_varobj_options.flat_output)
            .SetOmitSummaryDepth(m_varobj_options.no_summary_depth)
            .SetIgnoreCap(m_varobj_options.ignore_cap)
            .SetSummary(summary_format_sp);

        if (m_varobj_options.be_raw)
            options.SetRawDisplay(true);
        
        if (variable_list)
        {
            const Format format = m_option_format.GetFormat();
            options.SetFormat(format);

            if (command.GetArgumentCount() > 0)
            {
                VariableList regex_var_list;

                // If we have any args to the variable command, we will make
                // variable objects from them...
                for (idx = 0; (name_cstr = command.GetArgumentAtIndex(idx)) != NULL; ++idx)
                {
                    if (m_option_variable.use_regex)
                    {
                        const uint32_t regex_start_index = regex_var_list.GetSize();
                        RegularExpression regex (name_cstr);
                        if (regex.Compile(name_cstr))
                        {
                            size_t num_matches = 0;
                            const size_t num_new_regex_vars = variable_list->AppendVariablesIfUnique(regex, 
                                                                                                     regex_var_list, 
                                                                                                     num_matches);
                            if (num_new_regex_vars > 0)
                            {
                                for (uint32_t regex_idx = regex_start_index, end_index = regex_var_list.GetSize(); 
                                     regex_idx < end_index;
                                     ++regex_idx)
                                {
                                    var_sp = regex_var_list.GetVariableAtIndex (regex_idx);
                                    if (var_sp)
                                    {
                                        valobj_sp = frame->GetValueObjectForFrameVariable (var_sp, m_varobj_options.use_dynamic);
                                        if (valobj_sp)
                                        {
//                                            if (format != eFormatDefault)
//                                                valobj_sp->SetFormat (format);
                                            
                                            if (m_option_variable.show_decl && var_sp->GetDeclaration ().GetFile())
                                            {
                                                bool show_fullpaths = false;
                                                bool show_module = true;
                                                if (var_sp->DumpDeclaration(&s, show_fullpaths, show_module))
                                                    s.PutCString (": ");
                                            }
                                            ValueObject::DumpValueObject (result.GetOutputStream(), 
                                                                          valobj_sp.get(),
                                                                          options);
                                        }
                                    }
                                }
                            }
                            else if (num_matches == 0)
                            {
                                result.GetErrorStream().Printf ("error: no variables matched the regular expression '%s'.\n", name_cstr);
                            }
                        }
                        else
                        {
                            char regex_error[1024];
                            if (regex.GetErrorAsCString(regex_error, sizeof(regex_error)))
                                result.GetErrorStream().Printf ("error: %s\n", regex_error);
                            else
                                result.GetErrorStream().Printf ("error: unkown regex error when compiling '%s'\n", name_cstr);
                        }
                    }
                    else // No regex, either exact variable names or variable expressions.
                    {
                        Error error;
                        uint32_t expr_path_options = StackFrame::eExpressionPathOptionCheckPtrVsMember |
                                                     StackFrame::eExpressionPathOptionsAllowDirectIVarAccess;
                        lldb::VariableSP var_sp;
                        valobj_sp = frame->GetValueForVariableExpressionPath (name_cstr, 
                                                                              m_varobj_options.use_dynamic, 
                                                                              expr_path_options,
                                                                              var_sp,
                                                                              error);
                        if (valobj_sp)
                        {
//                            if (format != eFormatDefault)
//                                valobj_sp->SetFormat (format);
                            if (m_option_variable.show_decl && var_sp && var_sp->GetDeclaration ().GetFile())
                            {
                                var_sp->GetDeclaration ().DumpStopContext (&s, false);
                                s.PutCString (": ");
                            }
                            
                            options.SetFormat(format);

                            Stream &output_stream = result.GetOutputStream();
                            options.SetRootValueObjectName(valobj_sp->GetParent() ? name_cstr : NULL);
                            ValueObject::DumpValueObject (output_stream, 
                                                          valobj_sp.get(), 
                                                          options);
                        }
                        else
                        {
                            const char *error_cstr = error.AsCString(NULL);
                            if (error_cstr)
                                result.GetErrorStream().Printf("error: %s\n", error_cstr);
                            else
                                result.GetErrorStream().Printf ("error: unable to find any variable expression path that matches '%s'\n", name_cstr);
                        }
                    }
                }
            }
            else // No command arg specified.  Use variable_list, instead.
            {
                const uint32_t num_variables = variable_list->GetSize();
                if (num_variables > 0)
                {
                    for (uint32_t i=0; i<num_variables; i++)
                    {
                        var_sp = variable_list->GetVariableAtIndex(i);
                        bool dump_variable = true;
                        switch (var_sp->GetScope())
                        {
                            case eValueTypeVariableGlobal:
                                dump_variable = m_option_variable.show_globals;
                                if (dump_variable && m_option_variable.show_scope)
                                    s.PutCString("GLOBAL: ");
                                break;

                            case eValueTypeVariableStatic:
                                dump_variable = m_option_variable.show_globals;
                                if (dump_variable && m_option_variable.show_scope)
                                    s.PutCString("STATIC: ");
                                break;

                            case eValueTypeVariableArgument:
                                dump_variable = m_option_variable.show_args;
                                if (dump_variable && m_option_variable.show_scope)
                                    s.PutCString("   ARG: ");
                                break;

                            case eValueTypeVariableLocal:
                                dump_variable = m_option_variable.show_locals;
                                if (dump_variable && m_option_variable.show_scope)
                                    s.PutCString(" LOCAL: ");
                                break;

                            default:
                                break;
                        }

                        if (dump_variable)
                        {
                            // Use the variable object code to make sure we are
                            // using the same APIs as the the public API will be
                            // using...
                            valobj_sp = frame->GetValueObjectForFrameVariable (var_sp, 
                                                                               m_varobj_options.use_dynamic);
                            if (valobj_sp)
                            {
//                                if (format != eFormatDefault)
//                                    valobj_sp->SetFormat (format);

                                // When dumping all variables, don't print any variables
                                // that are not in scope to avoid extra unneeded output
                                if (valobj_sp->IsInScope ())
                                {
                                    if (m_option_variable.show_decl && var_sp->GetDeclaration ().GetFile())
                                    {
                                        var_sp->GetDeclaration ().DumpStopContext (&s, false);
                                        s.PutCString (": ");
                                    }
                                    
                                    options.SetFormat(format);
                                    options.SetRootValueObjectName(name_cstr);
                                    ValueObject::DumpValueObject (result.GetOutputStream(), 
                                                                  valobj_sp.get(), 
                                                                  options);
                                }
                            }
                        }
                    }
                }
            }
            result.SetStatus (eReturnStatusSuccessFinishResult);
        }
        
        if (m_interpreter.TruncationWarningNecessary())
        {
            result.GetOutputStream().Printf(m_interpreter.TruncationWarningText(),
                                            m_cmd_name.c_str());
            m_interpreter.TruncationWarningGiven();
        }
        
        return result.Succeeded();
    }
protected:

    OptionGroupOptions m_option_group;
    OptionGroupVariable m_option_variable;
    OptionGroupFormat m_option_format;
    OptionGroupValueObjectDisplay m_varobj_options;
};


#pragma mark CommandObjectMultiwordFrame

//-------------------------------------------------------------------------
// CommandObjectMultiwordFrame
//-------------------------------------------------------------------------

CommandObjectMultiwordFrame::CommandObjectMultiwordFrame (CommandInterpreter &interpreter) :
    CommandObjectMultiword (interpreter,
                            "frame",
                            "A set of commands for operating on the current thread's frames.",
                            "frame <subcommand> [<subcommand-options>]")
{
    LoadSubCommand ("info",   CommandObjectSP (new CommandObjectFrameInfo (interpreter)));
    LoadSubCommand ("select", CommandObjectSP (new CommandObjectFrameSelect (interpreter)));
    LoadSubCommand ("variable", CommandObjectSP (new CommandObjectFrameVariable (interpreter)));
}

CommandObjectMultiwordFrame::~CommandObjectMultiwordFrame ()
{
}

