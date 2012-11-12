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
#include "lldb/Host/Host.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/LineTable.h"
#include "lldb/Symbol/LineEntry.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
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

class CommandObjectThreadBacktrace : public CommandObjectParsed
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

        virtual
        ~CommandOptions ()
        {
        }

        virtual Error
        SetOptionValue (uint32_t option_idx, const char *option_arg)
        {
            Error error;
            char short_option = (char) m_getopt_table[option_idx].val;

            switch (short_option)
            {
                case 'c':
                {
                    bool success;
                    int32_t input_count =  Args::StringToSInt32 (option_arg, -1, 0, &success);
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
                    m_start =  Args::StringToUInt32 (option_arg, 0, 0, &success);
                    if (!success)
                        error.SetErrorStringWithFormat("invalid integer value for option '%c'", short_option);
                }
                break;
                default:
                    error.SetErrorStringWithFormat("invalid short option character '%c'", short_option);
                    break;

            }
            return error;
        }

        void
        OptionParsingStarting ()
        {
            m_count = UINT32_MAX;
            m_start = 0;
        }

        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }

        // Options table: Required for subclasses of Options.

        static OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.
        uint32_t m_count;
        uint32_t m_start;
    };

    CommandObjectThreadBacktrace (CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter,
                           "thread backtrace",
                           "Show the stack for one or more threads.  If no threads are specified, show the currently selected thread.  Use the thread-index \"all\" to see all threads.",
                           NULL,
                           eFlagProcessMustBeLaunched | eFlagProcessMustBePaused),
        m_options(interpreter)
    {
        CommandArgumentEntry arg;
        CommandArgumentData thread_idx_arg;
        
        // Define the first (and only) variant of this arg.
        thread_idx_arg.arg_type = eArgTypeThreadIndex;
        thread_idx_arg.arg_repetition = eArgRepeatStar;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg.push_back (thread_idx_arg);
        
        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg);
    }

    ~CommandObjectThreadBacktrace()
    {
    }

    virtual Options *
    GetOptions ()
    {
        return &m_options;
    }

protected:
    virtual bool
    DoExecute (Args& command, CommandReturnObject &result)
    {        
        result.SetStatus (eReturnStatusSuccessFinishResult);
        Stream &strm = result.GetOutputStream();

        // Don't show source context when doing backtraces.
        const uint32_t num_frames_with_source = 0;
        if (command.GetArgumentCount() == 0)
        {
            ExecutionContext exe_ctx(m_interpreter.GetExecutionContext());
            Thread *thread = exe_ctx.GetThreadPtr();
            if (thread)
            {
                // Thread::GetStatus() returns the number of frames shown.
                if (thread->GetStatus (strm,
                                       m_options.m_start,
                                       m_options.m_count,
                                       num_frames_with_source))
                {
                    result.SetStatus (eReturnStatusSuccessFinishResult);
                }
            }
            else
            {
                result.AppendError ("invalid thread");
                result.SetStatus (eReturnStatusFailed);
            }
        }
        else if (command.GetArgumentCount() == 1 && ::strcmp (command.GetArgumentAtIndex(0), "all") == 0)
        {
            Process *process = m_interpreter.GetExecutionContext().GetProcessPtr();
            Mutex::Locker locker (process->GetThreadList().GetMutex());
            uint32_t num_threads = process->GetThreadList().GetSize();
            for (uint32_t i = 0; i < num_threads; i++)
            {
                ThreadSP thread_sp = process->GetThreadList().GetThreadAtIndex(i);
                if (!thread_sp->GetStatus (strm,
                                           m_options.m_start,
                                           m_options.m_count,
                                           num_frames_with_source))
                {
                    result.AppendErrorWithFormat ("error displaying backtrace for thread: \"0x%4.4x\"\n", i);
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
                
                if (i < num_threads - 1)
                    result.AppendMessage("");
                    
            }
        }
        else
        {
            uint32_t num_args = command.GetArgumentCount();
            Process *process = m_interpreter.GetExecutionContext().GetProcessPtr();
            Mutex::Locker locker (process->GetThreadList().GetMutex());
            std::vector<ThreadSP> thread_sps;

            for (uint32_t i = 0; i < num_args; i++)
            {
                bool success;
                
                uint32_t thread_idx = Args::StringToUInt32(command.GetArgumentAtIndex(i), 0, 0, &success);
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
                if (!thread_sps[i]->GetStatus (strm,
                                               m_options.m_start,
                                               m_options.m_count,
                                               num_frames_with_source))
                {
                    result.AppendErrorWithFormat ("error displaying backtrace for thread: \"%s\"\n", command.GetArgumentAtIndex(i));
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
                
                if (i < num_args - 1)
                    result.AppendMessage("");
            }
        }
        return result.Succeeded();
    }

    CommandOptions m_options;
};

OptionDefinition
CommandObjectThreadBacktrace::CommandOptions::g_option_table[] =
{
{ LLDB_OPT_SET_1, false, "count", 'c', required_argument, NULL, 0, eArgTypeCount, "How many frames to display (-1 for all)"},
{ LLDB_OPT_SET_1, false, "start", 's', required_argument, NULL, 0, eArgTypeFrameIndex, "Frame in which to start the backtrace"},
{ 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
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

        virtual
        ~CommandOptions ()
        {
        }

        virtual Error
        SetOptionValue (uint32_t option_idx, const char *option_arg)
        {
            Error error;
            char short_option = (char) m_getopt_table[option_idx].val;

            switch (short_option)
            {
            case 'a':
                {
                    bool success;
                    m_avoid_no_debug =  Args::StringToBoolean (option_arg, true, &success);
                    if (!success)
                        error.SetErrorStringWithFormat("invalid boolean value for option '%c'", short_option);
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

            default:
                error.SetErrorStringWithFormat("invalid short option character '%c'", short_option);
                break;

            }
            return error;
        }

        void
        OptionParsingStarting ()
        {
            m_avoid_no_debug = true;
            m_run_mode = eOnlyDuringStepping;
            m_avoid_regexp.clear();
        }

        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }

        // Options table: Required for subclasses of Options.

        static OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.
        bool m_avoid_no_debug;
        RunMode m_run_mode;
        std::string m_avoid_regexp;
    };

    CommandObjectThreadStepWithTypeAndScope (CommandInterpreter &interpreter,
                                             const char *name,
                                             const char *help,
                                             const char *syntax,
                                             uint32_t flags,
                                             StepType step_type,
                                             StepScope step_scope) :
        CommandObjectParsed (interpreter, name, help, syntax, flags),
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

    virtual
    ~CommandObjectThreadStepWithTypeAndScope ()
    {
    }

    virtual
    Options *
    GetOptions ()
    {
        return &m_options;
    }

protected:
    virtual bool
    DoExecute (Args& command, CommandReturnObject &result)
    {
        Process *process = m_interpreter.GetExecutionContext().GetProcessPtr();
        bool synchronous_execution = m_interpreter.GetSynchronous();

        if (process == NULL)
        {
            result.AppendError ("need a valid process to step");
            result.SetStatus (eReturnStatusFailed);

        }
        else
        {
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
                uint32_t step_thread_idx = Args::StringToUInt32 (thread_idx_cstr, LLDB_INVALID_INDEX32);
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

            const bool abort_other_plans = false;
            const lldb::RunMode stop_other_threads = m_options.m_run_mode;
            
            // This is a bit unfortunate, but not all the commands in this command object support
            // only while stepping, so I use the bool for them.
            bool bool_stop_other_threads;
            if (m_options.m_run_mode == eAllThreads)
                bool_stop_other_threads = false;
            else if (m_options.m_run_mode == eOnlyDuringStepping)
            {
                if (m_step_type == eStepTypeOut)
                    bool_stop_other_threads = false;
                else
                    bool_stop_other_threads = true;
            }
            else
                bool_stop_other_threads = true;

            ThreadPlan *new_plan = NULL;
            
            if (m_step_type == eStepTypeInto)
            {
                StackFrame *frame = thread->GetStackFrameAtIndex(0).get();

                if (frame->HasDebugInformation ())
                {
                    new_plan = thread->QueueThreadPlanForStepRange (abort_other_plans, m_step_type, 
                                                                    frame->GetSymbolContext(eSymbolContextEverything).line_entry.range, 
                                                                    frame->GetSymbolContext(eSymbolContextEverything), 
                                                                    stop_other_threads,
                                                                    m_options.m_avoid_no_debug);
                    if (new_plan && !m_options.m_avoid_regexp.empty())
                    {
                        ThreadPlanStepInRange *step_in_range_plan = static_cast<ThreadPlanStepInRange *> (new_plan);
                        step_in_range_plan->SetAvoidRegexp(m_options.m_avoid_regexp.c_str());
                    }
                }
                else
                    new_plan = thread->QueueThreadPlanForStepSingleInstruction (false, abort_other_plans, bool_stop_other_threads);
                    
            }
            else if (m_step_type == eStepTypeOver)
            {
                StackFrame *frame = thread->GetStackFrameAtIndex(0).get();

                if (frame->HasDebugInformation())
                    new_plan = thread->QueueThreadPlanForStepRange (abort_other_plans, 
                                                                    m_step_type, 
                                                                    frame->GetSymbolContext(eSymbolContextEverything).line_entry.range, 
                                                                    frame->GetSymbolContext(eSymbolContextEverything), 
                                                                    stop_other_threads,
                                                                    false);
                else
                    new_plan = thread->QueueThreadPlanForStepSingleInstruction (true, 
                                                                                abort_other_plans, 
                                                                                bool_stop_other_threads);

            }
            else if (m_step_type == eStepTypeTrace)
            {
                new_plan = thread->QueueThreadPlanForStepSingleInstruction (false, abort_other_plans, bool_stop_other_threads);
            }
            else if (m_step_type == eStepTypeTraceOver)
            {
                new_plan = thread->QueueThreadPlanForStepSingleInstruction (true, abort_other_plans, bool_stop_other_threads);
            }
            else if (m_step_type == eStepTypeOut)
            {
                new_plan = thread->QueueThreadPlanForStepOut (abort_other_plans, 
                                                              NULL, 
                                                              false, 
                                                              bool_stop_other_threads, 
                                                              eVoteYes, 
                                                              eVoteNoOpinion, 
                                                              thread->GetSelectedFrameIndex());
            }
            else
            {
                result.AppendError ("step type is not supported");
                result.SetStatus (eReturnStatusFailed);
                return false;
            }
            
            // If we got a new plan, then set it to be a master plan (User level Plans should be master plans
            // so that they can be interruptible).  Then resume the process.
            
            if (new_plan != NULL)
            {
                new_plan->SetIsMasterPlan (true);
                new_plan->SetOkayToDiscard (false);

                process->GetThreadList().SetSelectedThreadByID (thread->GetID());
                process->Resume ();
            

                if (synchronous_execution)
                {
                    StateType state = process->WaitForProcessToStop (NULL);
                    
                    //EventSP event_sp;
                    //StateType state = process->WaitForStateChangedEvents (NULL, event_sp);
                    //while (! StateIsStoppedState (state))
                    //  {
                    //    state = process->WaitForStateChangedEvents (NULL, event_sp);
                    //  }
                    process->GetThreadList().SetSelectedThreadByID (thread->GetID());
                    result.SetDidChangeProcessState (true);
                    result.AppendMessageWithFormat ("Process %llu %s\n", process->GetID(), StateAsCString (state));
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
{ LLDB_OPT_SET_1, false, "avoid-no-debug",  'a', required_argument, NULL,               0, eArgTypeBoolean,     "A boolean value that sets whether step-in will step over functions with no debug information."},
{ LLDB_OPT_SET_1, false, "run-mode",        'm', required_argument, g_tri_running_mode, 0, eArgTypeRunMode, "Determine how to run other threads while stepping the current thread."},
{ LLDB_OPT_SET_1, false, "step-over-regexp",'r', required_argument, NULL,               0, eArgTypeRegularExpression,   "A regular expression that defines function names to step over."},
{ 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
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
                             eFlagProcessMustBeLaunched | eFlagProcessMustBePaused)
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


    virtual
    ~CommandObjectThreadContinue ()
    {
    }

    virtual bool
    DoExecute (Args& command, CommandReturnObject &result)
    {
        bool synchronous_execution = m_interpreter.GetSynchronous ();

        if (!m_interpreter.GetDebugger().GetSelectedTarget().get())
        {
            result.AppendError ("invalid target, create a debug target using the 'target create' command");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        Process *process = m_interpreter.GetExecutionContext().GetProcessPtr();
        if (process == NULL)
        {
            result.AppendError ("no process exists. Cannot continue");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        StateType state = process->GetState();
        if ((state == eStateCrashed) || (state == eStateStopped) || (state == eStateSuspended))
        {
            Mutex::Locker locker (process->GetThreadList().GetMutex());
            const uint32_t num_threads = process->GetThreadList().GetSize();
            const size_t argc = command.GetArgumentCount();
            if (argc > 0)
            {
                std::vector<Thread *> resume_threads;
                for (uint32_t i=0; i<argc; ++i)
                {
                    bool success;
                    const int base = 0;
                    uint32_t thread_idx = Args::StringToUInt32 (command.GetArgumentAtIndex(i), LLDB_INVALID_INDEX32, base, &success);
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
                            
                            thread->SetResumeState (eStateRunning);
                        }
                        else
                        {
                            thread->SetResumeState (eStateSuspended);
                        }
                    }
                    result.AppendMessageWithFormat ("in process %llu\n", process->GetID());
                }
            }
            else
            {
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
                        result.AppendMessageWithFormat ("Resuming thread 0x%4.4llx in process %llu\n", thread->GetID(), process->GetID());
                        thread->SetResumeState (eStateRunning);
                    }
                    else
                    {
                        thread->SetResumeState (eStateSuspended);
                    }
                }
            }
            
            Error error (process->Resume());
            if (error.Success())
            {
                result.AppendMessageWithFormat ("Process %llu resuming\n", process->GetID());
                if (synchronous_execution)
                {
                    state = process->WaitForProcessToStop (NULL);
                    
                    result.SetDidChangeProcessState (true);
                    result.AppendMessageWithFormat ("Process %llu %s\n", process->GetID(), StateAsCString (state));
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

        virtual
        ~CommandOptions ()
        {
        }

        virtual Error
        SetOptionValue (uint32_t option_idx, const char *option_arg)
        {
            Error error;
            char short_option = (char) m_getopt_table[option_idx].val;

            switch (short_option)
            {
                case 't':
                {
                    m_thread_idx = Args::StringToUInt32 (option_arg, LLDB_INVALID_INDEX32);
                    if (m_thread_idx == LLDB_INVALID_INDEX32)
                    {
                        error.SetErrorStringWithFormat ("invalid thread index '%s'", option_arg);
                    }
                }
                break;
                case 'f':
                {
                    m_frame_idx = Args::StringToUInt32 (option_arg, LLDB_INVALID_FRAME_ID);
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
        OptionParsingStarting ()
        {
            m_thread_idx = LLDB_INVALID_THREAD_ID;
            m_frame_idx = 0;
            m_stop_others = false;
        }

        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }

        uint32_t m_step_thread_idx;
        bool m_stop_others;

        // Options table: Required for subclasses of Options.

        static OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.
    };

    CommandObjectThreadUntil (CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter, 
                             "thread until",
                             "Run the current or specified thread until it reaches a given line number or leaves the current function.",
                             NULL,
                             eFlagProcessMustBeLaunched | eFlagProcessMustBePaused),
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


    virtual
    ~CommandObjectThreadUntil ()
    {
    }

    virtual
    Options *
    GetOptions ()
    {
        return &m_options;
    }

protected:
    virtual bool
    DoExecute (Args& command, CommandReturnObject &result)
    {
        bool synchronous_execution = m_interpreter.GetSynchronous ();

        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();
        if (target == NULL)
        {
            result.AppendError ("invalid target, create a debug target using the 'target create' command");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        Process *process = m_interpreter.GetExecutionContext().GetProcessPtr();
        if (process == NULL)
        {
            result.AppendError ("need a valid process to step");
            result.SetStatus (eReturnStatusFailed);

        }
        else
        {
            Thread *thread = NULL;
            uint32_t line_number;

            if (command.GetArgumentCount() != 1)
            {
                result.AppendErrorWithFormat ("No line number provided:\n%s", GetSyntax());
                result.SetStatus (eReturnStatusFailed);
                return false;
            }

            line_number = Args::StringToUInt32 (command.GetArgumentAtIndex(0), UINT32_MAX);
            if (line_number == UINT32_MAX)
            {
                result.AppendErrorWithFormat ("invalid line number: '%s'.\n", command.GetArgumentAtIndex(0));
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

            ThreadPlan *new_plan = NULL;

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
                line_table->FindLineEntryByAddress (fun_end_addr, function_start, &end_ptr);

                bool all_in_function = true;
                
                while (index_ptr <= end_ptr)
                {
                    LineEntry line_entry;
                    const bool exact = false;
                    index_ptr = sc.comp_unit->FindLineEntry(index_ptr, line_number, sc.comp_unit, exact, &line_entry);
                    if (index_ptr == UINT32_MAX)
                        break;

                    addr_t address = line_entry.range.GetBaseAddress().GetLoadAddress(target);
                    if (address != LLDB_INVALID_ADDRESS)
                    {
                        if (fun_addr_range.ContainsLoadAddress (address, target))
                            address_list.push_back (address);
                        else
                            all_in_function = false;
                    }
                    index_ptr++;
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
                
                new_plan = thread->QueueThreadPlanForStepUntil (abort_other_plans, 
                                                                &address_list.front(), 
                                                                address_list.size(), 
                                                                m_options.m_stop_others, 
                                                                m_options.m_frame_idx);
                // User level plans should be master plans so they can be interrupted (e.g. by hitting a breakpoint)
                // and other plans executed by the user (stepping around the breakpoint) and then a "continue"
                // will resume the original plan.
                new_plan->SetIsMasterPlan (true);
                new_plan->SetOkayToDiscard(false);
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
            Error error (process->Resume ());
            if (error.Success())
            {
                result.AppendMessageWithFormat ("Process %llu resuming\n", process->GetID());
                if (synchronous_execution)
                {
                    StateType state = process->WaitForProcessToStop (NULL);

                    result.SetDidChangeProcessState (true);
                    result.AppendMessageWithFormat ("Process %llu %s\n", process->GetID(), StateAsCString (state));
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
{ LLDB_OPT_SET_1, false, "frame",   'f', required_argument, NULL,               0, eArgTypeFrameIndex,   "Frame index for until operation - defaults to 0"},
{ LLDB_OPT_SET_1, false, "thread",  't', required_argument, NULL,               0, eArgTypeThreadIndex,  "Thread index for the thread for until operation"},
{ LLDB_OPT_SET_1, false, "run-mode",'m', required_argument, g_duo_running_mode, 0, eArgTypeRunMode,"Determine how to run other threads while stepping this one"},
{ 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
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
                             eFlagProcessMustBeLaunched | eFlagProcessMustBePaused)
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


    virtual
    ~CommandObjectThreadSelect ()
    {
    }

protected:
    virtual bool
    DoExecute (Args& command, CommandReturnObject &result)
    {
        Process *process = m_interpreter.GetExecutionContext().GetProcessPtr();
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

        uint32_t index_id = Args::StringToUInt32(command.GetArgumentAtIndex(0), 0, 0);

        Thread *new_thread = process->GetThreadList().FindThreadByIndexID(index_id).get();
        if (new_thread == NULL)
        {
            result.AppendErrorWithFormat ("invalid thread #%s.\n", command.GetArgumentAtIndex(0));
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        process->GetThreadList().SetSelectedThreadByID(new_thread->GetID());
        result.SetStatus (eReturnStatusSuccessFinishNoResult);
        
        const uint32_t start_frame = 0;
        const uint32_t num_frames = 1;
        const uint32_t num_frames_with_source = 1;
        new_thread->GetStatus (result.GetOutputStream(), 
                               start_frame,
                               num_frames,
                               num_frames_with_source);

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
                             eFlagProcessMustBeLaunched | eFlagProcessMustBePaused)
    {
    }

    ~CommandObjectThreadList()
    {
    }

protected:
    bool
    DoExecute (Args& command, CommandReturnObject &result)
    {
        Stream &strm = result.GetOutputStream();
        result.SetStatus (eReturnStatusSuccessFinishNoResult);
        ExecutionContext exe_ctx(m_interpreter.GetExecutionContext());
        Process *process = exe_ctx.GetProcessPtr();
        if (process)
        {
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
        }
        else
        {
            result.AppendError ("no current location or status available");
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }
};

class CommandObjectThreadReturn : public CommandObjectRaw
{
public:
    CommandObjectThreadReturn (CommandInterpreter &interpreter) :
        CommandObjectRaw (interpreter,
                          "thread return",
                          "Return from the currently selected frame, short-circuiting execution of the frames below it, with an optional return value.",
                          "thread return",
                          eFlagProcessMustBeLaunched | eFlagProcessMustBePaused)
    {
        CommandArgumentEntry arg;
        CommandArgumentData expression_arg;

        // Define the first (and only) variant of this arg.
        expression_arg.arg_type = eArgTypeExpression;
        expression_arg.arg_repetition = eArgRepeatPlain;

        // There is only one variant this argument could be; put it into the argument entry.
        arg.push_back (expression_arg);

        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg);
        
        
    }
    
    ~CommandObjectThreadReturn()
    {
    }
    
protected:

    bool DoExecute
    (
        const char *command,
        CommandReturnObject &result
    )
    {
        // If there is a command string, pass it to the expression parser:
        ExecutionContext exe_ctx = m_interpreter.GetExecutionContext();
        if (!(exe_ctx.HasProcessScope() && exe_ctx.HasThreadScope() && exe_ctx.HasFrameScope()))
        {
            result.AppendError("Must have selected process, thread and frame for thread return.");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        
        ValueObjectSP return_valobj_sp;
        
        StackFrameSP frame_sp = exe_ctx.GetFrameSP();
        uint32_t frame_idx = frame_sp->GetFrameIndex();
        
        if (frame_sp->IsInlined())
        {
            result.AppendError("Don't know how to return from inlined frames.");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        
        if (command && command[0] != '\0')
        {
            Target *target = exe_ctx.GetTargetPtr();
            EvaluateExpressionOptions options;

            options.SetUnwindOnError(true);
            options.SetUseDynamic(eNoDynamicValues);
            
            ExecutionResults exe_results = eExecutionSetupError;
            exe_results = target->EvaluateExpression (command,
                                                      frame_sp.get(),
                                                      return_valobj_sp,
                                                      options);
            if (exe_results != eExecutionCompleted)
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
        ThreadSP thread_sp = exe_ctx.GetThreadSP();
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
    LoadSubCommand ("select",     CommandObjectSP (new CommandObjectThreadSelect (interpreter)));
    LoadSubCommand ("until",      CommandObjectSP (new CommandObjectThreadUntil (interpreter)));
    LoadSubCommand ("step-in",    CommandObjectSP (new CommandObjectThreadStepWithTypeAndScope (
                                                    interpreter,
                                                    "thread step-in",
                                                    "Source level single step in specified thread (current thread, if none specified).",
                                                    NULL,
                                                    eFlagProcessMustBeLaunched | eFlagProcessMustBePaused,
                                                    eStepTypeInto,
                                                    eStepScopeSource)));
    
    LoadSubCommand ("step-out",   CommandObjectSP (new CommandObjectThreadStepWithTypeAndScope (
                                                    interpreter,
                                                    "thread step-out",
                                                    "Finish executing the function of the currently selected frame and return to its call site in specified thread (current thread, if none specified).",
                                                    NULL,
                                                    eFlagProcessMustBeLaunched | eFlagProcessMustBePaused,
                                                    eStepTypeOut,
                                                    eStepScopeSource)));

    LoadSubCommand ("step-over",   CommandObjectSP (new CommandObjectThreadStepWithTypeAndScope (
                                                    interpreter,
                                                    "thread step-over",
                                                    "Source level single step in specified thread (current thread, if none specified), stepping over calls.",
                                                    NULL,
                                                    eFlagProcessMustBeLaunched | eFlagProcessMustBePaused,
                                                    eStepTypeOver,
                                                    eStepScopeSource)));

    LoadSubCommand ("step-inst",   CommandObjectSP (new CommandObjectThreadStepWithTypeAndScope (
                                                    interpreter,
                                                    "thread step-inst",
                                                    "Single step one instruction in specified thread (current thread, if none specified).",
                                                    NULL,
                                                    eFlagProcessMustBeLaunched | eFlagProcessMustBePaused,
                                                    eStepTypeTrace,
                                                    eStepScopeInstruction)));

    LoadSubCommand ("step-inst-over", CommandObjectSP (new CommandObjectThreadStepWithTypeAndScope (
                                                    interpreter,
                                                    "thread step-inst-over",
                                                    "Single step one instruction in specified thread (current thread, if none specified), stepping over calls.",
                                                    NULL,
                                                    eFlagProcessMustBeLaunched | eFlagProcessMustBePaused,
                                                    eStepTypeTraceOver,
                                                    eStepScopeInstruction)));
}

CommandObjectMultiwordThread::~CommandObjectMultiwordThread ()
{
}


