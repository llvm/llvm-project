//===-- ThreadPlanCallFunction.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ThreadPlanCallFunction_h_
#define liblldb_ThreadPlanCallFunction_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlan.h"

namespace lldb_private {

class ThreadPlanCallFunction : public ThreadPlan
{
    // Create a thread plan to call a function at the address passed in the "function"
    // argument.  If you plan to call GetReturnValueObject, then pass in the 
    // return type, otherwise just pass in an invalid ClangASTType.
public:
    ThreadPlanCallFunction (Thread &thread,
                            const Address &function,
                            const ClangASTType &return_type,
                            lldb::addr_t arg,
                            bool stop_other_threads,
                            bool unwind_on_error = true,
                            bool ignore_breakpoints = false,
                            lldb::addr_t *this_arg = 0,
                            lldb::addr_t *cmd_arg = 0);

    ThreadPlanCallFunction (Thread &thread,
                            const Address &function,
                            const ClangASTType &return_type,
                            bool stop_other_threads,
                            bool unwind_on_error,
                            bool ignore_breakpoints,
                            lldb::addr_t *arg1_ptr = NULL,
                            lldb::addr_t *arg2_ptr = NULL,
                            lldb::addr_t *arg3_ptr = NULL,
                            lldb::addr_t *arg4_ptr = NULL,
                            lldb::addr_t *arg5_ptr = NULL,
                            lldb::addr_t *arg6_ptr = NULL);

    virtual
    ~ThreadPlanCallFunction ();

    virtual void
    GetDescription (Stream *s, lldb::DescriptionLevel level);

    virtual bool
    ValidatePlan (Stream *error);

    virtual bool
    ShouldStop (Event *event_ptr);
    
    virtual Vote
    ShouldReportStop(Event *event_ptr);

    virtual bool
    StopOthers ();
    
    virtual void
    SetStopOthers (bool new_value);

    virtual lldb::StateType
    GetPlanRunState ();

    virtual void
    DidPush ();

    virtual bool
    WillStop ();

    virtual bool
    MischiefManaged ();

    // To get the return value from a function call you must create a 
    // lldb::ValueSP that contains a valid clang type in its context and call
    // RequestReturnValue. The ValueSP will be stored and when the function is
    // done executing, the object will check if there is a requested return 
    // value. If there is, the return value will be retrieved using the 
    // ABI::GetReturnValue() for the ABI in the process. Then after the thread
    // plan is complete, you can call "GetReturnValue()" to retrieve the value
    // that was extracted.

    virtual lldb::ValueObjectSP
    GetReturnValueObject ()
    {
        return m_return_valobj_sp;
    }

    // Return the stack pointer that the function received
    // on entry.  Any stack address below this should be 
    // considered invalid after the function has been
    // cleaned up.
    lldb::addr_t
    GetFunctionStackPointer()
    {
        return m_function_sp;
    }
    
    // Classes that derive from ClangFunction, and implement
    // their own WillPop methods should call this so that the
    // thread state gets restored if the plan gets discarded.
    virtual void
    WillPop ();
    
    // If the thread plan stops mid-course, this will be the stop reason that interrupted us.
    // Once DoTakedown is called, this will be the real stop reason at the end of the function call.
    // If it hasn't been set for one or the other of these reasons, we'll return the PrivateStopReason.
    // This is needed because we want the CallFunction thread plans not to show up as the stop reason.
    // But if something bad goes wrong, it is nice to be able to tell the user what really happened.

    virtual lldb::StopInfoSP
    GetRealStopInfo()
    {
        if (m_real_stop_info_sp)
            return m_real_stop_info_sp;
        else
            return GetPrivateStopInfo ();
    }
    
    lldb::addr_t
    GetStopAddress ()
    {
        return m_stop_address;
    }

    virtual bool
    RestoreThreadState();

protected:    
    void ReportRegisterState (const char *message);

    virtual bool
    DoPlanExplainsStop (Event *event_ptr);

private:

    bool
    ConstructorSetup (Thread &thread,
                      ABI *& abi,
                      lldb::addr_t &start_load_addr,
                      lldb::addr_t &function_load_addr);

    void
    DoTakedown (bool success);
    
    void
    SetBreakpoints ();
    
    void
    ClearBreakpoints ();
    
    bool
    BreakpointsExplainStop ();
    
    bool                                            m_valid;
    bool                                            m_stop_other_threads;
    Address                                         m_function_addr;
    Address                                         m_start_addr;
    lldb::addr_t                                    m_function_sp;
    Thread::RegisterCheckpoint                      m_register_backup;
    lldb::ThreadPlanSP                              m_subplan_sp;
    LanguageRuntime                                *m_cxx_language_runtime;
    LanguageRuntime                                *m_objc_language_runtime;
    Thread::ThreadStateCheckpoint                   m_stored_thread_state;
    lldb::StopInfoSP                                m_real_stop_info_sp; // In general we want to hide call function
                                                                         // thread plans, but for reporting purposes,
                                                                         // it's nice to know the real stop reason.
                                                                         // This gets set in DoTakedown.
    StreamString                                    m_constructor_errors;
    ClangASTType                                    m_return_type;
    lldb::ValueObjectSP                             m_return_valobj_sp;  // If this contains a valid pointer, use the ABI to extract values when complete
    bool                                            m_takedown_done;    // We want to ensure we only do the takedown once.  This ensures that.
    lldb::addr_t                                    m_stop_address;     // This is the address we stopped at.  Also set in DoTakedown;
    bool                                            m_unwind_on_error;
    bool                                            m_ignore_breakpoints;

    DISALLOW_COPY_AND_ASSIGN (ThreadPlanCallFunction);
};

} // namespace lldb_private

#endif  // liblldb_ThreadPlanCallFunction_h_
