//===-- CommandObjectThread.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectThread.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Core/State.h"
#include "lldb/Core/SourceManager.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/StringConvert.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/LineTable.h"
#include "lldb/Symbol/LineEntry.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/SystemRuntime.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlan.h"
#include "lldb/Target/ThreadPlanStepInstruction.h"
#include "lldb/Target/ThreadPlanStepOut.h"
#include "lldb/Target/ThreadPlanStepRange.h"
#include "lldb/Target/ThreadPlanStepInRange.h"


using namespace lldb;
using namespace lldb_private;


//-------------------------------------------------------------------------
// CommandObjectThreadBacktrace
//-------------------------------------------------------------------------

class CommandObjectIterateOverThreads : public CommandObjectParsed
{
public:
    CommandObjectIterateOverThreads (CommandInterpreter &interpreter,
                         const char *name,
                         const char *help,
                         const char *syntax,
                         uint32_t flags) :
        CommandObjectParsed (interpreter, name, help, syntax, flags)
    {
    }

    ~CommandObjectIterateOverThreads() override {}

    bool
    DoExecute (Args& command, CommandReturnObject &result) override
    {        
        result.SetStatus (m_success_return);

        if (command.GetArgumentCount() == 0)
        {
            Thread *thread = m_exe_ctx.GetThreadPtr();
            if (!HandleOneThread (*thread, result))
                return false;
        }
        else if (command.GetArgumentCount() == 1 && ::strcmp (command.GetArgumentAtIndex(0), "all") == 0)
        {
            Process *process = m_exe_ctx.GetProcessPtr();
            uint32_t idx = 0;
            for (ThreadSP thread_sp : process->Threads())
            {
                if (idx != 0 && m_add_return)
                    result.AppendMessage("");

                if (!HandleOneThread(*(thread_sp.get()), result))
                    return false;
                ++idx;
            }
        }
        else
        {
            const size_t num_args = command.GetArgumentCount();
            Process *process = m_exe_ctx.GetProcessPtr();
            Mutex::Locker locker (process->GetThreadList().GetMutex());
            std::vector<ThreadSP> thread_sps;

            for (size_t i = 0; i < num_args; i++)
            {
                bool success;
                
                uint32_t thread_idx = StringConvert::ToUInt32(command.GetArgumentAtIndex(i), 0, 0, &success);
                if (!success)
                {
                    result.AppendErrorWithFormat ("invalid thread specification: \"%s\"\n", command.GetArgumentAtIndex(i));
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
                
                thread_sps.push_back(process->GetThreadList().FindThreadByIndexID(thread_idx));
                
                if (!thread_sps[i])
                {
                    result.AppendErrorWithFormat ("no thread with index: \"%s\"\n", command.GetArgumentAtIndex(i));
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
                
            }
            
            for (uint32_t i = 0; i < num_args; i++)
            {
                if (!HandleOneThread (*(thread_sps[i].get()), result))
                    return false;

                if (i < num_args - 1 && m_add_return)
                    result.AppendMessage("");
            }
        }
        return result.Succeeded();
    }

protected:

    // Override this to do whatever you need to do for one thread.
    //
    // If you return false, the iteration will stop, otherwise it will proceed.
    // The result is set to m_success_return (defaults to eReturnStatusSuccessFinishResult) before the iteration,
    // so you only need to set the return status in HandleOneThread if you want to indicate an error.
    // If m_add_return is true, a blank line will be inserted between each of the listings (except the last one.)

    virtual bool
    HandleOneThread (Thread &thread, CommandReturnObject &result) = 0;

    ReturnStatus m_success_return = eReturnStatusSuccessFinishResult;
    bool m_add_return = true;

};

//-------------------------------------------------------------------------
// CommandObjectThreadBacktrace
//-------------------------------------------------------------------------

class CommandObjectThreadBacktrace : public CommandObjectIterateOverThreads
{
public:

    class CommandOptions : public Options
    {
    public:

        CommandOptions (CommandInterpreter &interpreter) :
            Options(interpreter)
        {
            // Keep default values of all options in one place: OptionParsingStarting ()
            OptionParsingStarting ();
        }

        ~CommandOptions () override
        {
        }

        Error
        SetOptionValue (uint32_t option_idx, const char *option_arg) override
        {
            Error error;
            const int short_option = m_getopt_table[option_idx].val;

            switch (short_option)
            {
                case 'c':
                {
                    bool success;
                    int32_t input_count =  StringConvert::ToSInt32 (option_arg, -1, 0, &success);
                    if (!success)
                        error.SetErrorStringWithFormat("invalid integer value for option '%c'", short_option);
                    if (input_count < -1)
                        m_count = UINT32_MAX;
                    else
                        m_count = input_count;
                }
                break;
                case 's':
                {
                    bool success;
                    m_start =  StringConvert::ToUInt32 (option_arg, 0, 0, &success);
                    if (!success)
                        error.SetErrorStringWithFormat("invalid integer value for option '%c'", short_option);
                }
                case 'e':
                {
                    bool success;
                    m_extended_backtrace =  Args::StringToBoolean (option_arg, false, &success);
                    if (!success)
                        error.SetErrorStringWithFormat("invalid boolean value for option '%c'", short_option);
                }
                break;
                default:
                    error.SetErrorStringWithFormat("invalid short option character '%c'", short_option);
                    break;

            }
            return error;
        }

        void
        OptionParsingStarting () override
        {
            m_count = UINT32_MAX;
            m_start = 0;
            m_extended_backtrace = false;
        }

        const OptionDefinition*
        GetDefinitions () override
        {
            return g_option_table;
        }

        // Options table: Required for subclasses of Options.

        static OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.
        uint32_t m_count;
        uint32_t m_start;
        bool     m_extended_backtrace;
    };

    CommandObjectThreadBacktrace (CommandInterpreter &interpreter) :
        CommandObjectIterateOverThreads (interpreter,
                             "thread backtrace",
                             "Show the stack for one or more threads.  If no threads are specified, show the currently selected thread.  Use the thread-index \"all\" to see all threads.",
                             NULL,
                             eCommandRequiresProcess       |
                             eCommandRequiresThread        |
                             eCommandTryTargetAPILock      |
                             eCommandProcessMustBeLaunched |
                             eCommandProcessMustBePaused   ),
        m_options(interpreter)
    {
    }

    ~CommandObjectThreadBacktrace() override
    {
    }

    Options *
    GetOptions () override
    {
        return &m_options;
    }

protected:
    void
    DoExtendedBacktrace (Thread *thread, CommandReturnObject &result)
    {
        SystemRuntime *runtime = thread->GetProcess()->GetSystemRuntime();
        if (runtime)
        {
            Stream &strm = result.GetOutputStream();
            const std::vector<ConstString> &types = runtime->GetExtendedBacktraceTypes();
            for (auto type : types)
            {
                ThreadSP ext_thread_sp = runtime->GetExtendedBacktraceThread (thread->shared_from_this(), type);
                if (ext_thread_sp && ext_thread_sp->IsValid ())
                {
                    const uint32_t num_frames_with_source = 0;
                    if (ext_thread_sp->GetStatus (strm, 
                        m_options.m_start, 
                        m_options.m_count, 
                        num_frames_with_source))
                    {
                        DoExtendedBacktrace (ext_thread_sp.get(), result);
                    }
                }
            }
        }
    }

    bool
    HandleOneThread (Thread &thread, CommandReturnObject &result) override
    {
        Stream &strm = result.GetOutputStream();

        // Don't show source context when doing backtraces.
        const uint32_t num_frames_with_source = 0;

        if (!thread.GetStatus (strm,
                                   m_options.m_start,
                                   m_options.m_count,
                                   num_frames_with_source))
        {
            result.AppendErrorWithFormat ("error displaying backtrace for thread: \"0x%4.4x\"\n", thread.GetIndexID());
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        if (m_options.m_extended_backtrace)
        {
            DoExtendedBacktrace (&thread, result);
        }

        return true;
    }

    CommandOptions m_options;
};

OptionDefinition
CommandObjectThreadBacktrace::CommandOptions::g_option_table[] =
{
{ LLDB_OPT_SET_1, false, "count", 'c', OptionParser::eRequiredArgument, NULL, NULL, 0, eArgTypeCount, "How many frames to display (-1 for all)"},
{ LLDB_OPT_SET_1, false, "start", 's', OptionParser::eRequiredArgument, NULL, NULL, 0, eArgTypeFrameIndex, "Frame in which to start the backtrace"},
{ LLDB_OPT_SET_1, false, "extended", 'e', OptionParser::eRequiredArgument, NULL, NULL, 0, eArgTypeBoolean, "Show the extended backtrace, if available"},
{ 0, false, NULL, 0, 0, NULL, NULL, 0, eArgTypeNone, NULL }
};

enum StepScope
{
    eStepScopeSource,
    eStepScopeInstruction
};

class CommandObjectThreadStepWithTypeAndScope : public CommandObjectParsed
{
public:

    class CommandOptions : public Options
    {
    public:

        CommandOptions (CommandInterpreter &interpreter) :
            Options (interpreter)
        {
            // Keep default values of all options in one place: OptionParsingStarting ()
            OptionParsingStarting ();
        }

        ~CommandOptions () override
        {
        }

        Error
        SetOptionValue (uint32_t option_idx, const char *option_arg) override
        {
            Error error;
            const int short_option = m_getopt_table[option_idx].val;

            switch (short_option)
            {
            case 'a':
                {
                    bool success;
                    bool avoid_no_debug =  Args::StringToBoolean (option_arg, true, &success);
                    if (!success)
                        error.SetErrorStringWithFormat("invalid boolean value for option '%c'", short_option);
                    else
                    {
                        m_step_in_avoid_no_debug = avoid_no_debug ? eLazyBoolYes : eLazyBoolNo;
                    }
                }
                break;
            
            case 'A':
                {
                    bool success;
                    bool avoid_no_debug =  Args::StringToBoolean (option_arg, true, &success);
                    if (!success)
                        error.SetErrorStringWithFormat("invalid boolean value for option '%c'", short_option);
                    else
                    {
                        m_step_out_avoid_no_debug = avoid_no_debug ? eLazyBoolYes : eLazyBoolNo;
                    }
                }
                break;
            
            case 'c':
                {
                    m_step_count = StringConvert::ToUInt32(option_arg, UINT32_MAX, 0);
                    if (m_step_count == UINT32_MAX)
                       error.SetErrorStringWithFormat ("invalid ignore count '%s'", option_arg);
                    break;
                }
                break;
            case 'C':
                {
                    m_class_name.clear();
                    m_class_name.assign(option_arg);
                }
                break;
            case 'm':
                {
                    OptionEnumValueElement *enum_values = g_option_table[option_idx].enum_values; 
                    m_run_mode = (lldb::RunMode) Args::StringToOptionEnum(option_arg, enum_values, eOnlyDuringStepping, error);
                }
                break;
            
            case 'r':
                {
                    m_avoid_regexp.clear();
                    m_avoid_regexp.assign(option_arg);
                }
                break;

            case 't':
                {
                    m_step_in_target.clear();
                    m_step_in_target.assign(option_arg);

                }
                break;
            default:
                error.SetErrorStringWithFormat("invalid short option character '%c'", short_option);
                break;

            }
            return error;
        }

        void
        OptionParsingStarting () override
        {
            m_step_in_avoid_no_debug = eLazyBoolCalculate;
            m_step_out_avoid_no_debug = eLazyBoolCalculate;
            m_run_mode = eOnlyDuringStepping;

            // Check if we are in Non-Stop mode
            lldb::TargetSP target_sp = m_interpreter.GetDebugger().GetSelectedTarget();
            if (target_sp.get() != nullptr && target_sp->GetNonStopModeEnabled())
                m_run_mode = eOnlyThisThread;

            m_avoid_regexp.clear();
            m_step_in_target.clear();
            m_class_name.clear();
            m_step_count = 1;
        }

        const OptionDefinition*
        GetDefinitions () override
        {
            return g_option_table;
        }

        // Options table: Required for subclasses of Options.

        static OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.
        LazyBool m_step_in_avoid_no_debug;
        LazyBool m_step_out_avoid_no_debug;
        RunMode m_run_mode;
        std::string m_avoid_regexp;
        std::string m_step_in_target;
        std::string m_class_name;
        uint32_t m_step_count;
    };

    CommandObjectThreadStepWithTypeAndScope (CommandInterpreter &interpreter,
                                             const char *name,
                                             const char *help,
                                             const char *syntax,
                                             StepType step_type,
                                             StepScope step_scope) :
        CommandObjectParsed (interpreter, name, help, syntax,
                             eCommandRequiresProcess       |
                             eCommandRequiresThread        |
                             eCommandTryTargetAPILock      |
                             eCommandProcessMustBeLaunched |
                             eCommandProcessMustBePaused   ),
        m_step_type (step_type),
        m_step_scope (step_scope),
        m_options (interpreter)
    {
        CommandArgumentEntry arg;
        CommandArgumentData thread_id_arg;
        
        // Define the first (and only) variant of this arg.
        thread_id_arg.arg_type = eArgTypeThreadID;
        thread_id_arg.arg_repetition = eArgRepeatOptional;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg.push_back (thread_id_arg);
        
        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg);
    }

    ~CommandObjectThreadStepWithTypeAndScope () override
    {
    }

    Options *
    GetOptions () override
    {
        return &m_options;
    }

protected:
    bool
    DoExecute (Args& command, CommandReturnObject &result) override
    {
        Process *process = m_exe_ctx.GetProcessPtr();
        bool synchronous_execution = m_interpreter.GetSynchronous();

        const uint32_t num_threads = process->GetThreadList().GetSize();
        Thread *thread = NULL;

        if (command.GetArgumentCount() == 0)
        {
            thread = process->GetThreadList().GetSelectedThread().get();
            if (thread == NULL)
            {
                result.AppendError ("no selected thread in process");
                result.SetStatus (eReturnStatusFailed);
                return false;
            }
        }
        else
        {
            const char *thread_idx_cstr = command.GetArgumentAtIndex(0);
            uint32_t step_thread_idx = StringConvert::ToUInt32 (thread_idx_cstr, LLDB_INVALID_INDEX32);
            if (step_thread_idx == LLDB_INVALID_INDEX32)
            {
                result.AppendErrorWithFormat ("invalid thread index '%s'.\n", thread_idx_cstr);
                result.SetStatus (eReturnStatusFailed);
                return false;
            }
            thread = process->GetThreadList().FindThreadByIndexID(step_thread_idx).get();
            if (thread == NULL)
            {
                result.AppendErrorWithFormat ("Thread index %u is out of range (valid values are 0 - %u).\n", 
                                              step_thread_idx, num_threads);
                result.SetStatus (eReturnStatusFailed);
                return false;
            }
        }

        if (m_step_type == eStepTypeScripted)
        {
            if (m_options.m_class_name.empty())
            {
                result.AppendErrorWithFormat ("empty class name for scripted step.");
                result.SetStatus(eReturnStatusFailed);
                return false;
            }
            else if (!m_interpreter.GetScriptInterpreter()->CheckObjectExists(m_options.m_class_name.c_str()))
            {
                result.AppendErrorWithFormat ("class for scripted step: \"%s\" does not exist.", m_options.m_class_name.c_str());
                result.SetStatus(eReturnStatusFailed);
                return false;
            }
        }

        const bool abort_other_plans = false;
        const lldb::RunMode stop_other_threads = m_options.m_run_mode;
        
        // This is a bit unfortunate, but not all the commands in this command object support
        // only while stepping, so I use the bool for them.
        bool bool_stop_other_threads;
        if (m_options.m_run_mode == eAllThreads)
            bool_stop_other_threads = false;
        else if (m_options.m_run_mode == eOnlyDuringStepping)
        {
            if (m_step_type == eStepTypeOut || m_step_type == eStepTypeScripted)
                bool_stop_other_threads = false;
            else
                bool_stop_other_threads = true;
        }
        else
            bool_stop_other_threads = true;

        ThreadPlanSP new_plan_sp;
        
        if (m_step_type == eStepTypeInto)
        {
            StackFrame *frame = thread->GetStackFrameAtIndex(0).get();
            assert(frame != nullptr);

            if (frame->HasDebugInformation ())
            {
                new_plan_sp = thread->QueueThreadPlanForStepInRange (abort_other_plans,
                                                                frame->GetSymbolContext(eSymbolContextEverything).line_entry,
                                                                frame->GetSymbolContext(eSymbolContextEverything),
                                                                m_options.m_step_in_target.c_str(),
                                                                stop_other_threads,
                                                                m_options.m_step_in_avoid_no_debug,
                                                                m_options.m_step_out_avoid_no_debug);
                
                if (new_plan_sp && !m_options.m_avoid_regexp.empty())
                {
                    ThreadPlanStepInRange *step_in_range_plan = static_cast<ThreadPlanStepInRange *> (new_plan_sp.get());
                    step_in_range_plan->SetAvoidRegexp(m_options.m_avoid_regexp.c_str());
                }
            }
            else
                new_plan_sp = thread->QueueThreadPlanForStepSingleInstruction (false, abort_other_plans, bool_stop_other_threads);
                
        }
        else if (m_step_type == eStepTypeOver)
        {
            StackFrame *frame = thread->GetStackFrameAtIndex(0).get();

            if (frame->HasDebugInformation())
                new_plan_sp = thread->QueueThreadPlanForStepOverRange (abort_other_plans,
                                                                    frame->GetSymbolContext(eSymbolContextEverything).line_entry,
                                                                    frame->GetSymbolContext(eSymbolContextEverything), 
                                                                    stop_other_threads,
                                                                    m_options.m_step_out_avoid_no_debug);
            else
                new_plan_sp = thread->QueueThreadPlanForStepSingleInstruction (true,
                                                                            abort_other_plans, 
                                                                            bool_stop_other_threads);

        }
        else if (m_step_type == eStepTypeTrace)
        {
            new_plan_sp = thread->QueueThreadPlanForStepSingleInstruction (false, abort_other_plans, bool_stop_other_threads);
        }
        else if (m_step_type == eStepTypeTraceOver)
        {
            new_plan_sp = thread->QueueThreadPlanForStepSingleInstruction (true, abort_other_plans, bool_stop_other_threads);
        }
        else if (m_step_type == eStepTypeOut)
        {
            new_plan_sp = thread->QueueThreadPlanForStepOut (abort_other_plans,
                                                          NULL, 
                                                          false, 
                                                          bool_stop_other_threads, 
                                                          eVoteYes, 
                                                          eVoteNoOpinion, 
                                                          thread->GetSelectedFrameIndex(),
                                                          m_options.m_step_out_avoid_no_debug);
        }
        else if (m_step_type == eStepTypeScripted)
        {
            new_plan_sp = thread->QueueThreadPlanForStepScripted (abort_other_plans,
                                                                  m_options.m_class_name.c_str(),
                                                                  bool_stop_other_threads);
        }
        else
        {
            result.AppendError ("step type is not supported");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        
        // If we got a new plan, then set it to be a master plan (User level Plans should be master plans
        // so that they can be interruptible).  Then resume the process.
        
        if (new_plan_sp)
        {
            new_plan_sp->SetIsMasterPlan (true);
            new_plan_sp->SetOkayToDiscard (false);
            
            if (m_options.m_step_count > 1)
            {
                if (new_plan_sp->SetIterationCount(m_options.m_step_count))
                {
                    result.AppendWarning ("step operation does not support iteration count.");
                }
            }


            process->GetThreadList().SetSelectedThreadByID (thread->GetID());

            const uint32_t iohandler_id = process->GetIOHandlerID();

            StreamString stream;
            Error error;
            if (synchronous_execution)
                error = process->ResumeSynchronous (&stream);
            else
                error = process->Resume ();

            // There is a race condition where this thread will return up the call stack to the main command handler
            // and show an (lldb) prompt before HandlePrivateEvent (from PrivateStateThread) has
            // a chance to call PushProcessIOHandler().
            process->SyncIOHandler(iohandler_id, 2000);

            if (synchronous_execution)
            {
                // If any state changed events had anything to say, add that to the result
                if (stream.GetData())
                    result.AppendMessage(stream.GetData());

                process->GetThreadList().SetSelectedThreadByID (thread->GetID());
                result.SetDidChangeProcessState (true);
                result.SetStatus (eReturnStatusSuccessFinishNoResult);
            }
            else
            {
                result.SetStatus (eReturnStatusSuccessContinuingNoResult);
            }
        }
        else
        {
            result.AppendError ("Couldn't find thread plan to implement step type.");
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }

protected:
    StepType m_step_type;
    StepScope m_step_scope;
    CommandOptions m_options;
};

static OptionEnumValueElement
g_tri_running_mode[] =
{
{ eOnlyThisThread,     "this-thread",    "Run only this thread"},
{ eAllThreads,         "all-threads",    "Run all threads"},
{ eOnlyDuringStepping, "while-stepping", "Run only this thread while stepping"},
{ 0, NULL, NULL }
};

static OptionEnumValueElement
g_duo_running_mode[] =
{
{ eOnlyThisThread,     "this-thread",    "Run only this thread"},
{ eAllThreads,         "all-threads",    "Run all threads"},
{ 0, NULL, NULL }
};

OptionDefinition
CommandObjectThreadStepWithTypeAndScope::CommandOptions::g_option_table[] =
{
{ LLDB_OPT_SET_1, false, "step-in-avoids-no-debug",   'a', OptionParser::eRequiredArgument, NULL, NULL,               0, eArgTypeBoolean,     "A boolean value that sets whether stepping into functions will step over functions with no debug information."},
{ LLDB_OPT_SET_1, false, "step-out-avoids-no-debug",  'A', OptionParser::eRequiredArgument, NULL, NULL,               0, eArgTypeBoolean,     "A boolean value, if true stepping out of functions will continue to step out till it hits a function with debug information."},
{ LLDB_OPT_SET_1, false, "count",                     'c', OptionParser::eRequiredArgument, NULL, NULL,               1, eArgTypeCount,     "How many times to perform the stepping operation - currently only supported for step-inst and next-inst."},
{ LLDB_OPT_SET_1, false, "run-mode",                  'm', OptionParser::eRequiredArgument, NULL, g_tri_running_mode, 0, eArgTypeRunMode, "Determine how to run other threads while stepping the current thread."},
{ LLDB_OPT_SET_1, false, "step-over-regexp",          'r', OptionParser::eRequiredArgument, NULL, NULL,               0, eArgTypeRegularExpression,   "A regular expression that defines function names to not to stop at when stepping in."},
{ LLDB_OPT_SET_1, false, "step-in-target",            't', OptionParser::eRequiredArgument, NULL, NULL,               0, eArgTypeFunctionName,   "The name of the directly called function step in should stop at when stepping into."},
{ LLDB_OPT_SET_2, false, "python-class",              'C', OptionParser::eRequiredArgument, NULL, NULL,               0, eArgTypePythonClass, "The name of the class that will manage this step - only supported for Scripted Step."},
{ 0, false, NULL, 0, 0, NULL, NULL, 0, eArgTypeNone, NULL }
};


//-------------------------------------------------------------------------
// CommandObjectThreadContinue
//-------------------------------------------------------------------------

class CommandObjectThreadContinue : public CommandObjectParsed
{
public:

    CommandObjectThreadContinue (CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter, 
                             "thread continue",
                             "Continue execution of one or more threads in an active process.",
                             NULL,
                             eCommandRequiresThread        |
                             eCommandTryTargetAPILock      |
                             eCommandProcessMustBeLaunched |
                             eCommandProcessMustBePaused)
    {
        CommandArgumentEntry arg;
        CommandArgumentData thread_idx_arg;
        
        // Define the first (and only) variant of this arg.
        thread_idx_arg.arg_type = eArgTypeThreadIndex;
        thread_idx_arg.arg_repetition = eArgRepeatPlus;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg.push_back (thread_idx_arg);
        
        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg);
    }


    ~CommandObjectThreadContinue () override
    {
    }

    bool
    DoExecute (Args& command, CommandReturnObject &result) override
    {
        bool synchronous_execution = m_interpreter.GetSynchronous ();

        if (!m_interpreter.GetDebugger().GetSelectedTarget().get())
        {
            result.AppendError ("invalid target, create a debug target using the 'target create' command");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        Process *process = m_exe_ctx.GetProcessPtr();
        if (process == NULL)
        {
            result.AppendError ("no process exists. Cannot continue");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        StateType state = process->GetState();
        if ((state == eStateCrashed) || (state == eStateStopped) || (state == eStateSuspended))
        {
            const size_t argc = command.GetArgumentCount();
            if (argc > 0)
            {
                // These two lines appear at the beginning of both blocks in
                // this if..else, but that is because we need to release the
                // lock before calling process->Resume below.
                Mutex::Locker locker (process->GetThreadList().GetMutex());
                const uint32_t num_threads = process->GetThreadList().GetSize();
                std::vector<Thread *> resume_threads;
                for (uint32_t i=0; i<argc; ++i)
                {
                    bool success;
                    const int base = 0;
                    uint32_t thread_idx = StringConvert::ToUInt32 (command.GetArgumentAtIndex(i), LLDB_INVALID_INDEX32, base, &success);
                    if (success)
                    {
                        Thread *thread = process->GetThreadList().FindThreadByIndexID(thread_idx).get();

                        if (thread)
                        {
                            resume_threads.push_back(thread);
                        }
                        else
                        {
                            result.AppendErrorWithFormat("invalid thread index %u.\n", thread_idx);
                            result.SetStatus (eReturnStatusFailed);
                            return false;
                        }
                    }
                    else
                    {
                        result.AppendErrorWithFormat ("invalid thread index argument: \"%s\".\n", command.GetArgumentAtIndex(i));
                        result.SetStatus (eReturnStatusFailed);
                        return false;
                    }
                }

                if (resume_threads.empty())
                {
                    result.AppendError ("no valid thread indexes were specified");
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
                else
                {
                    if (resume_threads.size() == 1)
                        result.AppendMessageWithFormat ("Resuming thread: ");
                    else
                        result.AppendMessageWithFormat ("Resuming threads: ");

                    for (uint32_t idx=0; idx<num_threads; ++idx)
                    {
                        Thread *thread = process->GetThreadList().GetThreadAtIndex(idx).get();
                        std::vector<Thread *>::iterator this_thread_pos = find(resume_threads.begin(), resume_threads.end(), thread);

                        if (this_thread_pos != resume_threads.end())
                        {
                            resume_threads.erase(this_thread_pos);
                            if (resume_threads.size() > 0)
                                result.AppendMessageWithFormat ("%u, ", thread->GetIndexID());
                            else
                                result.AppendMessageWithFormat ("%u ", thread->GetIndexID());
                            
                            const bool override_suspend = true;
                            thread->SetResumeState (eStateRunning, override_suspend);
                        }
                        else
                        {
                            thread->SetResumeState (eStateSuspended);
                        }
                    }
                    result.AppendMessageWithFormat ("in process %" PRIu64 "\n", process->GetID());
                }
            }
            else
            {
                // These two lines appear at the beginning of both blocks in
                // this if..else, but that is because we need to release the
                // lock before calling process->Resume below.
                Mutex::Locker locker (process->GetThreadList().GetMutex());
                const uint32_t num_threads = process->GetThreadList().GetSize();
                Thread *current_thread = process->GetThreadList().GetSelectedThread().get();
                if (current_thread == NULL)
                {
                    result.AppendError ("the process doesn't have a current thread");
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
                // Set the actions that the threads should each take when resuming
                for (uint32_t idx=0; idx<num_threads; ++idx)
                {
                    Thread *thread = process->GetThreadList().GetThreadAtIndex(idx).get();
                    if (thread == current_thread)
                    {
                        result.AppendMessageWithFormat ("Resuming thread 0x%4.4" PRIx64 " in process %" PRIu64 "\n", thread->GetID(), process->GetID());
                        const bool override_suspend = true;
                        thread->SetResumeState (eStateRunning, override_suspend);
                    }
                    else
                    {
                        thread->SetResumeState (eStateSuspended);
                    }
                }
            }


            StreamString stream;
            Error error;
            if (synchronous_execution)
                error = process->ResumeSynchronous (&stream);
            else
                error = process->Resume ();

            // We should not be holding the thread list lock when we do this.
            if (error.Success())
            {
                result.AppendMessageWithFormat ("Process %" PRIu64 " resuming\n", process->GetID());
                if (synchronous_execution)
                {
                    // If any state changed events had anything to say, add that to the result
                    if (stream.GetData())
                        result.AppendMessage(stream.GetData());

                    result.SetDidChangeProcessState (true);
                    result.SetStatus (eReturnStatusSuccessFinishNoResult);
                }
                else
                {
                    result.SetStatus (eReturnStatusSuccessContinuingNoResult);
                }
            }
            else
            {
                result.AppendErrorWithFormat("Failed to resume process: %s\n", error.AsCString());
                result.SetStatus (eReturnStatusFailed);
            }
        }
        else
        {
            result.AppendErrorWithFormat ("Process cannot be continued from its current state (%s).\n",
                                          StateAsCString(state));
            result.SetStatus (eReturnStatusFailed);
        }

        return result.Succeeded();
    }

};

//-------------------------------------------------------------------------
// CommandObjectThreadUntil
//-------------------------------------------------------------------------

class CommandObjectThreadUntil : public CommandObjectParsed
{
public:

    class CommandOptions : public Options
    {
    public:
        uint32_t m_thread_idx;
        uint32_t m_frame_idx;

        CommandOptions (CommandInterpreter &interpreter) :
            Options (interpreter),
            m_thread_idx(LLDB_INVALID_THREAD_ID),
            m_frame_idx(LLDB_INVALID_FRAME_ID)
        {
            // Keep default values of all options in one place: OptionParsingStarting ()
            OptionParsingStarting ();
        }

        ~CommandOptions () override
        {
        }

        Error
        SetOptionValue (uint32_t option_idx, const char *option_arg) override
        {
            Error error;
            const int short_option = m_getopt_table[option_idx].val;

            switch (short_option)
            {
                case 'a':
                {
                    ExecutionContext exe_ctx (m_interpreter.GetExecutionContext());
                    lldb::addr_t tmp_addr = Args::StringToAddress(&exe_ctx, option_arg, LLDB_INVALID_ADDRESS, &error);
                    if (error.Success())
                        m_until_addrs.push_back(tmp_addr);
                }
                break;
                case 't':
                {
                    m_thread_idx = StringConvert::ToUInt32 (option_arg, LLDB_INVALID_INDEX32);
                    if (m_thread_idx == LLDB_INVALID_INDEX32)
                    {
                        error.SetErrorStringWithFormat ("invalid thread index '%s'", option_arg);
                    }
                }
                break;
                case 'f':
                {
                    m_frame_idx = StringConvert::ToUInt32 (option_arg, LLDB_INVALID_FRAME_ID);
                    if (m_frame_idx == LLDB_INVALID_FRAME_ID)
                    {
                        error.SetErrorStringWithFormat ("invalid frame index '%s'", option_arg);
                    }
                }
                break;
                case 'm':
                {
                    OptionEnumValueElement *enum_values = g_option_table[option_idx].enum_values; 
                    lldb::RunMode run_mode = (lldb::RunMode) Args::StringToOptionEnum(option_arg, enum_values, eOnlyDuringStepping, error);

                    if (error.Success())
                    {
                        if (run_mode == eAllThreads)
                            m_stop_others = false;
                        else
                            m_stop_others = true;
                    }
                }
                break;
                default:
                    error.SetErrorStringWithFormat("invalid short option character '%c'", short_option);
                    break;

            }
            return error;
        }

        void
        OptionParsingStarting () override
        {
            m_thread_idx = LLDB_INVALID_THREAD_ID;
            m_frame_idx = 0;
            m_stop_others = false;
            m_until_addrs.clear();
        }

        const OptionDefinition*
        GetDefinitions () override
        {
            return g_option_table;
        }

        uint32_t m_step_thread_idx;
        bool m_stop_others;
        std::vector<lldb::addr_t> m_until_addrs;

        // Options table: Required for subclasses of Options.

        static OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.
    };

    CommandObjectThreadUntil (CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter, 
                             "thread until",
                             "Run the current or specified thread until it reaches a given line number or address or leaves the current function.",
                             NULL,
                             eCommandRequiresThread        |
                             eCommandTryTargetAPILock      |
                             eCommandProcessMustBeLaunched |
                             eCommandProcessMustBePaused   ),
        m_options (interpreter)
    {
        CommandArgumentEntry arg;
        CommandArgumentData line_num_arg;
        
        // Define the first (and only) variant of this arg.
        line_num_arg.arg_type = eArgTypeLineNum;
        line_num_arg.arg_repetition = eArgRepeatPlain;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg.push_back (line_num_arg);
        
        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg);
    }


    ~CommandObjectThreadUntil () override
    {
    }

    Options *
    GetOptions () override
    {
        return &m_options;
    }

protected:
    bool
    DoExecute (Args& command, CommandReturnObject &result) override
    {
        bool synchronous_execution = m_interpreter.GetSynchronous ();

        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target == NULL)
        {
            result.AppendError ("invalid target, create a debug target using the 'target create' command");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        Process *process = m_exe_ctx.GetProcessPtr();
        if (process == NULL)
        {
            result.AppendError ("need a valid process to step");
            result.SetStatus (eReturnStatusFailed);

        }
        else
        {
            Thread *thread = NULL;
            std::vector<uint32_t> line_numbers;

            if (command.GetArgumentCount() >= 1)
            {
                size_t num_args = command.GetArgumentCount();
                for (size_t i = 0; i < num_args; i++)
                {
                    uint32_t line_number;
                    line_number = StringConvert::ToUInt32 (command.GetArgumentAtIndex(0), UINT32_MAX);
                    if (line_number == UINT32_MAX)
                    {
                        result.AppendErrorWithFormat ("invalid line number: '%s'.\n", command.GetArgumentAtIndex(0));
                        result.SetStatus (eReturnStatusFailed);
                        return false;
                    }
                    else
                        line_numbers.push_back(line_number);
                }
            }
            else if (m_options.m_until_addrs.empty())
            {
                result.AppendErrorWithFormat ("No line number or address provided:\n%s", GetSyntax());
                result.SetStatus (eReturnStatusFailed);
                return false;
            }


            if (m_options.m_thread_idx == LLDB_INVALID_THREAD_ID)
            {
                thread = process->GetThreadList().GetSelectedThread().get();
            }
            else
            {
                thread = process->GetThreadList().FindThreadByIndexID(m_options.m_thread_idx).get();
            }

            if (thread == NULL)
            {
                const uint32_t num_threads = process->GetThreadList().GetSize();
                result.AppendErrorWithFormat ("Thread index %u is out of range (valid values are 0 - %u).\n", 
                                              m_options.m_thread_idx, 
                                              num_threads);
                result.SetStatus (eReturnStatusFailed);
                return false;
            }

            const bool abort_other_plans = false;

            StackFrame *frame = thread->GetStackFrameAtIndex(m_options.m_frame_idx).get();
            if (frame == NULL)
            {

                result.AppendErrorWithFormat ("Frame index %u is out of range for thread %u.\n", 
                                              m_options.m_frame_idx, 
                                              m_options.m_thread_idx);
                result.SetStatus (eReturnStatusFailed);
                return false;
            }

            ThreadPlanSP new_plan_sp;

            if (frame->HasDebugInformation ())
            {
                // Finally we got here...  Translate the given line number to a bunch of addresses:
                SymbolContext sc(frame->GetSymbolContext (eSymbolContextCompUnit));
                LineTable *line_table = NULL;
                if (sc.comp_unit)
                    line_table = sc.comp_unit->GetLineTable();

                if (line_table == NULL)
                {
                    result.AppendErrorWithFormat ("Failed to resolve the line table for frame %u of thread index %u.\n",
                                                 m_options.m_frame_idx, m_options.m_thread_idx);
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }

                LineEntry function_start;
                uint32_t index_ptr = 0, end_ptr;
                std::vector<addr_t> address_list;

                // Find the beginning & end index of the
                AddressRange fun_addr_range = sc.function->GetAddressRange();
                Address fun_start_addr = fun_addr_range.GetBaseAddress();
                line_table->FindLineEntryByAddress (fun_start_addr, function_start, &index_ptr);

                Address fun_end_addr(fun_start_addr.GetSection(), 
                                     fun_start_addr.GetOffset() + fun_addr_range.GetByteSize());

                bool all_in_function = true;

                line_table->FindLineEntryByAddress (fun_end_addr, function_start, &end_ptr);

                for (uint32_t line_number : line_numbers)
                {
                    uint32_t start_idx_ptr = index_ptr;
                    while (start_idx_ptr <= end_ptr)
                    {
                        LineEntry line_entry;
                        const bool exact = false;
                        start_idx_ptr = sc.comp_unit->FindLineEntry(start_idx_ptr, line_number, sc.comp_unit, exact, &line_entry);
                        if (start_idx_ptr == UINT32_MAX)
                            break;

                        addr_t address = line_entry.range.GetBaseAddress().GetLoadAddress(target);
                        if (address != LLDB_INVALID_ADDRESS)
                        {
                            if (fun_addr_range.ContainsLoadAddress (address, target))
                                address_list.push_back (address);
                            else
                                all_in_function = false;
                        }
                        start_idx_ptr++;
                    }
                }

                for (lldb::addr_t address : m_options.m_until_addrs)
                {
                    if (fun_addr_range.ContainsLoadAddress (address, target))
                        address_list.push_back (address);
                    else
                        all_in_function = false;
                }

                if (address_list.size() == 0)
                {
                    if (all_in_function)
                        result.AppendErrorWithFormat ("No line entries matching until target.\n");
                    else
                        result.AppendErrorWithFormat ("Until target outside of the current function.\n");
                        
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
                
                new_plan_sp = thread->QueueThreadPlanForStepUntil (abort_other_plans,
                                                                &address_list.front(), 
                                                                address_list.size(), 
                                                                m_options.m_stop_others, 
                                                                m_options.m_frame_idx);
                // User level plans should be master plans so they can be interrupted (e.g. by hitting a breakpoint)
                // and other plans executed by the user (stepping around the breakpoint) and then a "continue"
                // will resume the original plan.
                new_plan_sp->SetIsMasterPlan (true);
                new_plan_sp->SetOkayToDiscard(false);
            }
            else
            {
                result.AppendErrorWithFormat ("Frame index %u of thread %u has no debug information.\n", 
                                              m_options.m_frame_idx, 
                                              m_options.m_thread_idx);
                result.SetStatus (eReturnStatusFailed);
                return false;

            }



            process->GetThreadList().SetSelectedThreadByID (m_options.m_thread_idx);

            StreamString stream;
            Error error;
            if (synchronous_execution)
                error = process->ResumeSynchronous (&stream);
            else
                error = process->Resume ();

            if (error.Success())
            {
                result.AppendMessageWithFormat ("Process %" PRIu64 " resuming\n", process->GetID());
                if (synchronous_execution)
                {
                    // If any state changed events had anything to say, add that to the result
                    if (stream.GetData())
                        result.AppendMessage(stream.GetData());

                    result.SetDidChangeProcessState (true);
                    result.SetStatus (eReturnStatusSuccessFinishNoResult);
                }
                else
                {
                    result.SetStatus (eReturnStatusSuccessContinuingNoResult);
                }
            }
            else
            {
                result.AppendErrorWithFormat("Failed to resume process: %s.\n", error.AsCString());
                result.SetStatus (eReturnStatusFailed);
            }

        }
        return result.Succeeded();
    }

    CommandOptions m_options;

};

OptionDefinition
CommandObjectThreadUntil::CommandOptions::g_option_table[] =
{
{ LLDB_OPT_SET_1, false, "frame",   'f', OptionParser::eRequiredArgument, NULL, NULL,               0, eArgTypeFrameIndex,   "Frame index for until operation - defaults to 0"},
{ LLDB_OPT_SET_1, false, "thread",  't', OptionParser::eRequiredArgument, NULL, NULL,               0, eArgTypeThreadIndex,  "Thread index for the thread for until operation"},
{ LLDB_OPT_SET_1, false, "run-mode",'m', OptionParser::eRequiredArgument, NULL, g_duo_running_mode, 0, eArgTypeRunMode, "Determine how to run other threads while stepping this one"},
{ LLDB_OPT_SET_1, false, "address", 'a', OptionParser::eRequiredArgument, NULL, NULL,               0, eArgTypeAddressOrExpression, "Run until we reach the specified address, or leave the function - can be specified multiple times."},
{ 0, false, NULL, 0, 0, NULL, NULL, 0, eArgTypeNone, NULL }
};


//-------------------------------------------------------------------------
// CommandObjectThreadSelect
//-------------------------------------------------------------------------

class CommandObjectThreadSelect : public CommandObjectParsed
{
public:

    CommandObjectThreadSelect (CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter,
                             "thread select",
                             "Select a thread as the currently active thread.",
                             NULL,
                             eCommandRequiresProcess       |
                             eCommandTryTargetAPILock      |
                             eCommandProcessMustBeLaunched |
                             eCommandProcessMustBePaused   )
    {
        CommandArgumentEntry arg;
        CommandArgumentData thread_idx_arg;
        
        // Define the first (and only) variant of this arg.
        thread_idx_arg.arg_type = eArgTypeThreadIndex;
        thread_idx_arg.arg_repetition = eArgRepeatPlain;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg.push_back (thread_idx_arg);
        
        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg);
    }


    ~CommandObjectThreadSelect () override
    {
    }

protected:
    bool
    DoExecute (Args& command, CommandReturnObject &result) override
    {
        Process *process = m_exe_ctx.GetProcessPtr();
        if (process == NULL)
        {
            result.AppendError ("no process");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        else if (command.GetArgumentCount() != 1)
        {
            result.AppendErrorWithFormat("'%s' takes exactly one thread index argument:\nUsage: %s\n", m_cmd_name.c_str(), m_cmd_syntax.c_str());
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        uint32_t index_id = StringConvert::ToUInt32(command.GetArgumentAtIndex(0), 0, 0);

        Thread *new_thread = process->GetThreadList().FindThreadByIndexID(index_id).get();
        if (new_thread == NULL)
        {
            result.AppendErrorWithFormat ("invalid thread #%s.\n", command.GetArgumentAtIndex(0));
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        process->GetThreadList().SetSelectedThreadByID(new_thread->GetID(), true);
        result.SetStatus (eReturnStatusSuccessFinishNoResult);
        
        return result.Succeeded();
    }

};


//-------------------------------------------------------------------------
// CommandObjectThreadList
//-------------------------------------------------------------------------

class CommandObjectThreadList : public CommandObjectParsed
{
public:


    CommandObjectThreadList (CommandInterpreter &interpreter):
        CommandObjectParsed (interpreter,
                             "thread list",
                             "Show a summary of all current threads in a process.",
                             "thread list",
                             eCommandRequiresProcess       |
                             eCommandTryTargetAPILock      |
                             eCommandProcessMustBeLaunched |
                             eCommandProcessMustBePaused   )
    {
    }

    ~CommandObjectThreadList() override
    {
    }

protected:
    bool
    DoExecute (Args& command, CommandReturnObject &result) override
    {
        Stream &strm = result.GetOutputStream();
        result.SetStatus (eReturnStatusSuccessFinishNoResult);
        Process *process = m_exe_ctx.GetProcessPtr();
        const bool only_threads_with_stop_reason = false;
        const uint32_t start_frame = 0;
        const uint32_t num_frames = 0;
        const uint32_t num_frames_with_source = 0;
        process->GetStatus(strm);
        process->GetThreadStatus (strm, 
                                  only_threads_with_stop_reason, 
                                  start_frame,
                                  num_frames,
                                  num_frames_with_source);            
        return result.Succeeded();
    }
};

//-------------------------------------------------------------------------
// CommandObjectThreadInfo
//-------------------------------------------------------------------------

class CommandObjectThreadInfo : public CommandObjectIterateOverThreads
{
public:

    CommandObjectThreadInfo (CommandInterpreter &interpreter) :
        CommandObjectIterateOverThreads (interpreter,
                                         "thread info",
                                         "Show an extended summary of information about thread(s) in a process.",
                                         "thread info",
                                         eCommandRequiresProcess       |
                                         eCommandTryTargetAPILock      |
                                         eCommandProcessMustBeLaunched |
                                         eCommandProcessMustBePaused),
        m_options (interpreter)
    {
        m_add_return = false;
    }

    class CommandOptions : public Options
    {
    public:

        CommandOptions (CommandInterpreter &interpreter) :
            Options (interpreter)
        {
            OptionParsingStarting ();
        }

        void
        OptionParsingStarting () override
        {
            m_json_thread = false;
            m_json_stopinfo = false;
        }

        ~CommandOptions () override
        {
        }

        Error
        SetOptionValue (uint32_t option_idx, const char *option_arg) override
        {
            const int short_option = m_getopt_table[option_idx].val;
            Error error;

            switch (short_option)
            {
                case 'j':
                    m_json_thread = true;
                    break;
                    
                case 's':
                    m_json_stopinfo = true;
                    break;

                default:
                    return Error("invalid short option character '%c'", short_option);

            }
            return error;
        }

        const OptionDefinition*
        GetDefinitions () override
        {
            return g_option_table;
        }

        bool m_json_thread;
        bool m_json_stopinfo;

        static OptionDefinition g_option_table[];
    };

    Options *
    GetOptions () override
    {
        return &m_options;
    }

    ~CommandObjectThreadInfo () override
    {
    }

    bool
    HandleOneThread (Thread &thread, CommandReturnObject &result) override
    {
        Stream &strm = result.GetOutputStream();
        if (!thread.GetDescription (strm, eDescriptionLevelFull, m_options.m_json_thread, m_options.m_json_stopinfo))
        {
            result.AppendErrorWithFormat ("error displaying info for thread: \"%d\"\n", thread.GetIndexID());
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        return true;
    }

    CommandOptions m_options;

};

OptionDefinition
CommandObjectThreadInfo::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_ALL, false, "json",'j', OptionParser::eNoArgument, NULL, NULL, 0, eArgTypeNone, "Display the thread info in JSON format."},
    { LLDB_OPT_SET_ALL, false, "stop-info",'s', OptionParser::eNoArgument, NULL, NULL, 0, eArgTypeNone, "Display the extended stop info in JSON format."},

    { 0, false, NULL, 0, 0, NULL, NULL, 0, eArgTypeNone, NULL }
};


//-------------------------------------------------------------------------
// CommandObjectThreadReturn
//-------------------------------------------------------------------------

class CommandObjectThreadReturn : public CommandObjectRaw
{
public:
    class CommandOptions : public Options
    {
    public:

        CommandOptions (CommandInterpreter &interpreter) :
            Options (interpreter),
            m_from_expression (false)
        {
            // Keep default values of all options in one place: OptionParsingStarting ()
            OptionParsingStarting ();
        }

        ~CommandOptions () override
        {
        }

        Error
        SetOptionValue (uint32_t option_idx, const char *option_arg) override
        {
            Error error;
            const int short_option = m_getopt_table[option_idx].val;

            switch (short_option)
            {
                case 'x':
                {
                    bool success;
                    bool tmp_value = Args::StringToBoolean (option_arg, false, &success);
                    if (success)
                        m_from_expression = tmp_value;
                    else
                    {
                        error.SetErrorStringWithFormat ("invalid boolean value '%s' for 'x' option", option_arg);
                    }
                }
                break;
                default:
                    error.SetErrorStringWithFormat("invalid short option character '%c'", short_option);
                    break;

            }
            return error;
        }

        void
        OptionParsingStarting () override
        {
            m_from_expression = false;
        }

        const OptionDefinition*
        GetDefinitions () override
        {
            return g_option_table;
        }

        bool m_from_expression;

        // Options table: Required for subclasses of Options.

        static OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.
    };

    Options *
    GetOptions () override
    {
        return &m_options;
    }

    CommandObjectThreadReturn (CommandInterpreter &interpreter) :
        CommandObjectRaw (interpreter,
                          "thread return",
                          "Return from the currently selected frame, short-circuiting execution of the frames below it, with an optional return value,"
                          " or with the -x option from the innermost function evaluation.",
                          "thread return",
                          eCommandRequiresFrame         |
                          eCommandTryTargetAPILock      |
                          eCommandProcessMustBeLaunched |
                          eCommandProcessMustBePaused   ),
        m_options (interpreter)
    {
        CommandArgumentEntry arg;
        CommandArgumentData expression_arg;

        // Define the first (and only) variant of this arg.
        expression_arg.arg_type = eArgTypeExpression;
        expression_arg.arg_repetition = eArgRepeatOptional;

        // There is only one variant this argument could be; put it into the argument entry.
        arg.push_back (expression_arg);

        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg);
        
        
    }
    
    ~CommandObjectThreadReturn() override
    {
    }
    
protected:

    bool
    DoExecute (const char *command, CommandReturnObject &result) override
    {
        // I am going to handle this by hand, because I don't want you to have to say:
        // "thread return -- -5".
        if (command[0] == '-' && command[1] == 'x')
        {
            if (command && command[2] != '\0')
                result.AppendWarning("Return values ignored when returning from user called expressions");
            
            Thread *thread = m_exe_ctx.GetThreadPtr();
            Error error;
            error = thread->UnwindInnermostExpression();
            if (!error.Success())
            {
                result.AppendErrorWithFormat ("Unwinding expression failed - %s.", error.AsCString());
                result.SetStatus (eReturnStatusFailed);
            }
            else
            {
                bool success = thread->SetSelectedFrameByIndexNoisily (0, result.GetOutputStream());
                if (success)
                {
                    m_exe_ctx.SetFrameSP(thread->GetSelectedFrame ());
                    result.SetStatus (eReturnStatusSuccessFinishResult);
                }
                else
                {
                    result.AppendErrorWithFormat ("Could not select 0th frame after unwinding expression.");
                    result.SetStatus (eReturnStatusFailed);
                }
            }
            return result.Succeeded();
        }
        
        ValueObjectSP return_valobj_sp;
        
        StackFrameSP frame_sp = m_exe_ctx.GetFrameSP();
        uint32_t frame_idx = frame_sp->GetFrameIndex();
        
        if (frame_sp->IsInlined())
        {
            result.AppendError("Don't know how to return from inlined frames.");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        
        if (command && command[0] != '\0')
        {
            Target *target = m_exe_ctx.GetTargetPtr();
            EvaluateExpressionOptions options;

            options.SetUnwindOnError(true);
            options.SetUseDynamic(eNoDynamicValues);
            
            ExpressionResults exe_results = eExpressionSetupError;
            exe_results = target->EvaluateExpression (command,
                                                      frame_sp.get(),
                                                      return_valobj_sp,
                                                      options);
            if (exe_results != eExpressionCompleted)
            {
                if (return_valobj_sp)
                    result.AppendErrorWithFormat("Error evaluating result expression: %s", return_valobj_sp->GetError().AsCString());
                else
                    result.AppendErrorWithFormat("Unknown error evaluating result expression.");
                result.SetStatus (eReturnStatusFailed);
                return false;
            
            }
        }
                
        Error error;
        ThreadSP thread_sp = m_exe_ctx.GetThreadSP();
        const bool broadcast = true;
        error = thread_sp->ReturnFromFrame (frame_sp, return_valobj_sp, broadcast);
        if (!error.Success())
        {
            result.AppendErrorWithFormat("Error returning from frame %d of thread %d: %s.", frame_idx, thread_sp->GetIndexID(), error.AsCString());
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        result.SetStatus (eReturnStatusSuccessFinishResult);
        return true;
    }
    
    CommandOptions m_options;

};
OptionDefinition
CommandObjectThreadReturn::CommandOptions::g_option_table[] =
{
{ LLDB_OPT_SET_ALL, false, "from-expression",  'x', OptionParser::eNoArgument, NULL, NULL,               0, eArgTypeNone,     "Return from the innermost expression evaluation."},
{ 0, false, NULL, 0, 0, NULL, NULL, 0, eArgTypeNone, NULL }
};

//-------------------------------------------------------------------------
// CommandObjectThreadJump
//-------------------------------------------------------------------------

class CommandObjectThreadJump : public CommandObjectParsed
{
public:
    class CommandOptions : public Options
    {
    public:

        CommandOptions (CommandInterpreter &interpreter) :
            Options (interpreter)
        {
            OptionParsingStarting ();
        }

        void
        OptionParsingStarting () override
        {
            m_filenames.Clear();
            m_line_num = 0;
            m_line_offset = 0;
            m_load_addr = LLDB_INVALID_ADDRESS;
            m_force = false;
        }

        ~CommandOptions () override
        {
        }

        Error
        SetOptionValue (uint32_t option_idx, const char *option_arg) override
        {
            bool success;
            const int short_option = m_getopt_table[option_idx].val;
            Error error;

            switch (short_option)
            {
                case 'f':
                    m_filenames.AppendIfUnique (FileSpec(option_arg, false));
                    if (m_filenames.GetSize() > 1)
                        return Error("only one source file expected.");
                    break;
                case 'l':
                    m_line_num = StringConvert::ToUInt32 (option_arg, 0, 0, &success);
                    if (!success || m_line_num == 0)
                        return Error("invalid line number: '%s'.", option_arg);
                    break;
                case 'b':
                    m_line_offset = StringConvert::ToSInt32 (option_arg, 0, 0, &success);
                    if (!success)
                        return Error("invalid line offset: '%s'.", option_arg);
                    break;
                case 'a':
                    {
                        ExecutionContext exe_ctx (m_interpreter.GetExecutionContext());
                        m_load_addr = Args::StringToAddress(&exe_ctx, option_arg, LLDB_INVALID_ADDRESS, &error);
                    }
                    break;
                case 'r':
                    m_force = true;
                    break;

                 default:
                    return Error("invalid short option character '%c'", short_option);

            }
            return error;
        }

        const OptionDefinition*
        GetDefinitions () override
        {
            return g_option_table;
        }

        FileSpecList m_filenames;
        uint32_t m_line_num;
        int32_t m_line_offset;
        lldb::addr_t m_load_addr;
        bool m_force;

        static OptionDefinition g_option_table[];
    };

    Options *
    GetOptions () override
    {
        return &m_options;
    }

    CommandObjectThreadJump (CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter,
                          "thread jump",
                          "Sets the program counter to a new address.",
                          "thread jump",
                          eCommandRequiresFrame         |
                          eCommandTryTargetAPILock      |
                          eCommandProcessMustBeLaunched |
                          eCommandProcessMustBePaused   ),
        m_options (interpreter)
    {
    }

    ~CommandObjectThreadJump() override
    {
    }

protected:

    bool DoExecute (Args& args, CommandReturnObject &result) override
    {
        RegisterContext *reg_ctx = m_exe_ctx.GetRegisterContext();
        StackFrame *frame = m_exe_ctx.GetFramePtr();
        Thread *thread = m_exe_ctx.GetThreadPtr();
        Target *target = m_exe_ctx.GetTargetPtr();
        const SymbolContext &sym_ctx = frame->GetSymbolContext (eSymbolContextLineEntry);

        if (m_options.m_load_addr != LLDB_INVALID_ADDRESS)
        {
            // Use this address directly.
            Address dest = Address(m_options.m_load_addr);

            lldb::addr_t callAddr = dest.GetCallableLoadAddress (target);
            if (callAddr == LLDB_INVALID_ADDRESS)
            {
                result.AppendErrorWithFormat ("Invalid destination address.");
                result.SetStatus (eReturnStatusFailed);
                return false;
            }

            if (!reg_ctx->SetPC (callAddr))
            {
                result.AppendErrorWithFormat ("Error changing PC value for thread %d.", thread->GetIndexID());
                result.SetStatus (eReturnStatusFailed);
                return false;
            }
        }
        else
        {
            // Pick either the absolute line, or work out a relative one.
            int32_t line = (int32_t)m_options.m_line_num;
            if (line == 0)
                line = sym_ctx.line_entry.line + m_options.m_line_offset;

            // Try the current file, but override if asked.
            FileSpec file = sym_ctx.line_entry.file;
            if (m_options.m_filenames.GetSize() == 1)
                file = m_options.m_filenames.GetFileSpecAtIndex(0);

            if (!file)
            {
                result.AppendErrorWithFormat ("No source file available for the current location.");
                result.SetStatus (eReturnStatusFailed);
                return false;
            }

            std::string warnings;
            Error err = thread->JumpToLine (file, line, m_options.m_force, &warnings);

            if (err.Fail())
            {
                result.SetError (err);
                return false;
            }

            if (!warnings.empty())
                result.AppendWarning (warnings.c_str());
        }

        result.SetStatus (eReturnStatusSuccessFinishResult);
        return true;
    }

    CommandOptions m_options;
};
OptionDefinition
CommandObjectThreadJump::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_1, false, "file", 'f', OptionParser::eRequiredArgument, NULL, NULL, CommandCompletions::eSourceFileCompletion, eArgTypeFilename,
        "Specifies the source file to jump to."},

    { LLDB_OPT_SET_1, true, "line", 'l', OptionParser::eRequiredArgument, NULL, NULL, 0, eArgTypeLineNum,
        "Specifies the line number to jump to."},

    { LLDB_OPT_SET_2, true, "by", 'b', OptionParser::eRequiredArgument, NULL, NULL, 0, eArgTypeOffset,
        "Jumps by a relative line offset from the current line."},

    { LLDB_OPT_SET_3, true, "address", 'a', OptionParser::eRequiredArgument, NULL, NULL, 0, eArgTypeAddressOrExpression,
        "Jumps to a specific address."},

    { LLDB_OPT_SET_1|
      LLDB_OPT_SET_2|
      LLDB_OPT_SET_3, false, "force",'r', OptionParser::eNoArgument, NULL, NULL, 0, eArgTypeNone,"Allows the PC to leave the current function."},

    { 0, false, NULL, 0, 0, NULL, NULL, 0, eArgTypeNone, NULL }
};

//-------------------------------------------------------------------------
// Next are the subcommands of CommandObjectMultiwordThreadPlan
//-------------------------------------------------------------------------


//-------------------------------------------------------------------------
// CommandObjectThreadPlanList
//-------------------------------------------------------------------------
class CommandObjectThreadPlanList : public CommandObjectIterateOverThreads
{
public:

    class CommandOptions : public Options
    {
    public:

        CommandOptions (CommandInterpreter &interpreter) :
            Options(interpreter)
        {
            // Keep default values of all options in one place: OptionParsingStarting ()
            OptionParsingStarting ();
        }

        ~CommandOptions () override
        {
        }

        Error
        SetOptionValue (uint32_t option_idx, const char *option_arg) override
        {
            Error error;
            const int short_option = m_getopt_table[option_idx].val;

            switch (short_option)
            {
                case 'i':
                {
                    m_internal = true;
                }
                break;
                case 'v':
                {
                    m_verbose = true;
                }
                break;
                default:
                    error.SetErrorStringWithFormat("invalid short option character '%c'", short_option);
                    break;

            }
            return error;
        }

        void
        OptionParsingStarting () override
        {
            m_verbose = false;
            m_internal = false;
        }

        const OptionDefinition*
        GetDefinitions () override
        {
            return g_option_table;
        }

        // Options table: Required for subclasses of Options.

        static OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.
        bool m_verbose;
        bool m_internal;
    };

    CommandObjectThreadPlanList (CommandInterpreter &interpreter) :
        CommandObjectIterateOverThreads (interpreter,
                                         "thread plan list",
                                         "Show thread plans for one or more threads.  If no threads are specified, show the "
                                         "currently selected thread.  Use the thread-index \"all\" to see all threads.",
                                         NULL,
                                         eCommandRequiresProcess       |
                                         eCommandRequiresThread        |
                                         eCommandTryTargetAPILock      |
                                         eCommandProcessMustBeLaunched |
                                         eCommandProcessMustBePaused   ),
        m_options(interpreter)
    {
    }

    ~CommandObjectThreadPlanList () override
    {
    }

    Options *
    GetOptions () override
    {
        return &m_options;
    }

protected:
    bool
    HandleOneThread (Thread &thread, CommandReturnObject &result) override
    {
        Stream &strm = result.GetOutputStream();
        DescriptionLevel desc_level = eDescriptionLevelFull;
        if (m_options.m_verbose)
            desc_level = eDescriptionLevelVerbose;

        thread.DumpThreadPlans (&strm, desc_level, m_options.m_internal, true);
        return true;
    }
    CommandOptions m_options;
};

OptionDefinition
CommandObjectThreadPlanList::CommandOptions::g_option_table[] =
{
{ LLDB_OPT_SET_1, false, "verbose", 'v', OptionParser::eNoArgument, NULL, NULL, 0, eArgTypeNone, "Display more information about the thread plans"},
{ LLDB_OPT_SET_1, false, "internal", 'i', OptionParser::eNoArgument, NULL, NULL, 0, eArgTypeNone, "Display internal as well as user thread plans"},
{ 0, false, NULL, 0, 0, NULL, NULL, 0, eArgTypeNone, NULL }
};

class CommandObjectThreadPlanDiscard : public CommandObjectParsed
{
public:
    CommandObjectThreadPlanDiscard (CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter,
                             "thread plan discard",
                             "Discards thread plans up to and including the plan passed as the command argument."
                             "Only user visible plans can be discarded, use the index from \"thread plan list\""
                             " without the \"-i\" argument.",
                             NULL,
                             eCommandRequiresProcess       |
                             eCommandRequiresThread        |
                             eCommandTryTargetAPILock      |
                             eCommandProcessMustBeLaunched |
                             eCommandProcessMustBePaused   )
    {
        CommandArgumentEntry arg;
        CommandArgumentData plan_index_arg;

        // Define the first (and only) variant of this arg.
        plan_index_arg.arg_type = eArgTypeUnsignedInteger;
        plan_index_arg.arg_repetition = eArgRepeatPlain;

        // There is only one variant this argument could be; put it into the argument entry.
        arg.push_back (plan_index_arg);

        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg);
    }

    ~CommandObjectThreadPlanDiscard () override {}

    bool
    DoExecute (Args& args, CommandReturnObject &result) override
    {
        Thread *thread = m_exe_ctx.GetThreadPtr();
        if (args.GetArgumentCount() != 1)
        {
            result.AppendErrorWithFormat("Too many arguments, expected one - the thread plan index - but got %zu.",
                                         args.GetArgumentCount());
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        bool success;
        uint32_t thread_plan_idx = StringConvert::ToUInt32(args.GetArgumentAtIndex(0), 0, 0, &success);
        if (!success)
        {
            result.AppendErrorWithFormat("Invalid thread index: \"%s\" - should be unsigned int.",
                                         args.GetArgumentAtIndex(0));
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        if (thread_plan_idx == 0)
        {
            result.AppendErrorWithFormat("You wouldn't really want me to discard the base thread plan.");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        if (thread->DiscardUserThreadPlansUpToIndex(thread_plan_idx))
        {
            result.SetStatus(eReturnStatusSuccessFinishNoResult);
            return true;
        }
        else
        {
            result.AppendErrorWithFormat("Could not find User thread plan with index %s.",
                                         args.GetArgumentAtIndex(0));
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
    }
};

//-------------------------------------------------------------------------
// CommandObjectMultiwordThreadPlan
//-------------------------------------------------------------------------

class CommandObjectMultiwordThreadPlan : public CommandObjectMultiword
{
public:
    CommandObjectMultiwordThreadPlan(CommandInterpreter &interpreter) :
        CommandObjectMultiword (interpreter,
                                "plan",
                                "A set of subcommands for accessing the thread plans controlling execution control on one or more threads.",
                                "thread plan <subcommand> [<subcommand objects]")
    {
        LoadSubCommand ("list", CommandObjectSP (new CommandObjectThreadPlanList (interpreter)));
        LoadSubCommand ("discard", CommandObjectSP (new CommandObjectThreadPlanDiscard (interpreter)));
    }

    ~CommandObjectMultiwordThreadPlan () override {}


};

//-------------------------------------------------------------------------
// CommandObjectMultiwordThread
//-------------------------------------------------------------------------

CommandObjectMultiwordThread::CommandObjectMultiwordThread (CommandInterpreter &interpreter) :
    CommandObjectMultiword (interpreter,
                            "thread",
                            "A set of commands for operating on one or more threads within a running process.",
                            "thread <subcommand> [<subcommand-options>]")
{
    LoadSubCommand ("backtrace",  CommandObjectSP (new CommandObjectThreadBacktrace (interpreter)));
    LoadSubCommand ("continue",   CommandObjectSP (new CommandObjectThreadContinue (interpreter)));
    LoadSubCommand ("list",       CommandObjectSP (new CommandObjectThreadList (interpreter)));
    LoadSubCommand ("return",     CommandObjectSP (new CommandObjectThreadReturn (interpreter)));
    LoadSubCommand ("jump",       CommandObjectSP (new CommandObjectThreadJump (interpreter)));
    LoadSubCommand ("select",     CommandObjectSP (new CommandObjectThreadSelect (interpreter)));
    LoadSubCommand ("until",      CommandObjectSP (new CommandObjectThreadUntil (interpreter)));
    LoadSubCommand ("info",       CommandObjectSP (new CommandObjectThreadInfo (interpreter)));
    LoadSubCommand ("step-in",    CommandObjectSP (new CommandObjectThreadStepWithTypeAndScope (
                                                    interpreter,
                                                    "thread step-in",
                                                    "Source level single step in specified thread (current thread, if none specified).",
                                                    NULL,
                                                    eStepTypeInto,
                                                    eStepScopeSource)));
    
    LoadSubCommand ("step-out",   CommandObjectSP (new CommandObjectThreadStepWithTypeAndScope (
                                                    interpreter,
                                                    "thread step-out",
                                                    "Finish executing the function of the currently selected frame and return to its call site in specified thread (current thread, if none specified).",
                                                    NULL,
                                                    eStepTypeOut,
                                                    eStepScopeSource)));

    LoadSubCommand ("step-over",   CommandObjectSP (new CommandObjectThreadStepWithTypeAndScope (
                                                    interpreter,
                                                    "thread step-over",
                                                    "Source level single step in specified thread (current thread, if none specified), stepping over calls.",
                                                    NULL,
                                                    eStepTypeOver,
                                                    eStepScopeSource)));

    LoadSubCommand ("step-inst",   CommandObjectSP (new CommandObjectThreadStepWithTypeAndScope (
                                                    interpreter,
                                                    "thread step-inst",
                                                    "Single step one instruction in specified thread (current thread, if none specified).",
                                                    NULL,
                                                    eStepTypeTrace,
                                                    eStepScopeInstruction)));

    LoadSubCommand ("step-inst-over", CommandObjectSP (new CommandObjectThreadStepWithTypeAndScope (
                                                    interpreter,
                                                    "thread step-inst-over",
                                                    "Single step one instruction in specified thread (current thread, if none specified), stepping over calls.",
                                                    NULL,
                                                    eStepTypeTraceOver,
                                                    eStepScopeInstruction)));

    LoadSubCommand ("step-scripted", CommandObjectSP (new CommandObjectThreadStepWithTypeAndScope (
                                                    interpreter,
                                                    "thread step-scripted",
                                                    "Step as instructed by the script class passed in the -C option.",
                                                    NULL,
                                                    eStepTypeScripted,
                                                    eStepScopeSource)));

    LoadSubCommand ("plan", CommandObjectSP (new CommandObjectMultiwordThreadPlan(interpreter)));
}

CommandObjectMultiwordThread::~CommandObjectMultiwordThread ()
{
}


