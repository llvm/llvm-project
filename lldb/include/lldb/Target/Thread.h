//===-- Thread.h ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Thread_h_
#define liblldb_Thread_h_

#include "lldb/lldb-private.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Core/Broadcaster.h"
#include "lldb/Core/Event.h"
#include "lldb/Core/UserID.h"
#include "lldb/Core/UserSettingsController.h"
#include "lldb/Target/ExecutionContextScope.h"
#include "lldb/Target/StackFrameList.h"

#define LLDB_THREAD_MAX_STOP_EXC_DATA 8

namespace lldb_private {

class ThreadProperties : public Properties
{
public:
    ThreadProperties(bool is_global);
    
    virtual
    ~ThreadProperties();
    
    //------------------------------------------------------------------
    /// The regular expression returned determines symbols that this
    /// thread won't stop in during "step-in" operations.
    ///
    /// @return
    ///    A pointer to a regular expression to compare against symbols,
    ///    or NULL if all symbols are allowed.
    ///
    //------------------------------------------------------------------
    const RegularExpression *
    GetSymbolsToAvoidRegexp();
    
    bool
    GetTraceEnabledState() const;
};

typedef STD_SHARED_PTR(ThreadProperties) ThreadPropertiesSP;

class Thread :
    public STD_ENABLE_SHARED_FROM_THIS(Thread),
    public ThreadProperties,
    public UserID,
    public ExecutionContextScope,
    public Broadcaster
{
friend class ThreadEventData;

public:
    //------------------------------------------------------------------
    /// Broadcaster event bits definitions.
    //------------------------------------------------------------------
    enum
    {
        eBroadcastBitStackChanged           = (1 << 0),
        eBroadcastBitThreadSuspended        = (1 << 1),
        eBroadcastBitThreadResumed          = (1 << 2),
        eBroadcastBitSelectedFrameChanged  = (1 << 3)
    };

    static ConstString &GetStaticBroadcasterClass ();
    
    virtual ConstString &GetBroadcasterClass() const
    {
        return GetStaticBroadcasterClass();
    }
    
    class ThreadEventData :
        public EventData
    {
    public:
        ThreadEventData (const lldb::ThreadSP thread_sp);
        
        ThreadEventData (const lldb::ThreadSP thread_sp, const StackID &stack_id);
        
        ThreadEventData();
        
        virtual ~ThreadEventData();
        
        static const ConstString &
        GetFlavorString ();

        virtual const ConstString &
        GetFlavor () const
        {
            return ThreadEventData::GetFlavorString ();
        }
        
        virtual void
        Dump (Stream *s) const;
    
        static const ThreadEventData *
        GetEventDataFromEvent (const Event *event_ptr);
        
        static lldb::ThreadSP
        GetThreadFromEvent (const Event *event_ptr);
        
        static StackID
        GetStackIDFromEvent (const Event *event_ptr);
        
        static lldb::StackFrameSP
        GetStackFrameFromEvent (const Event *event_ptr);
        
        lldb::ThreadSP
        GetThread () const
        {
            return m_thread_sp;
        }
        
        StackID
        GetStackID () const
        {
            return m_stack_id;
        }
    
    private:
        lldb::ThreadSP m_thread_sp;
        StackID        m_stack_id;
    DISALLOW_COPY_AND_ASSIGN (ThreadEventData);
    };
    
    // TODO: You shouldn't just checkpoint the register state alone, so this should get
    // moved to protected.  To do that ThreadStateCheckpoint needs to be returned as a token...
    class RegisterCheckpoint
    {
    public:

        RegisterCheckpoint() :
            m_stack_id (),
            m_data_sp ()
        {
        }

        RegisterCheckpoint (const StackID &stack_id) :
            m_stack_id (stack_id),
            m_data_sp ()
        {
        }

        ~RegisterCheckpoint()
        {
        }

        const RegisterCheckpoint&
        operator= (const RegisterCheckpoint &rhs)
        {
            if (this != &rhs)
            {
                this->m_stack_id = rhs.m_stack_id;
                this->m_data_sp  = rhs.m_data_sp;
            }
            return *this;
        }
        
        RegisterCheckpoint (const RegisterCheckpoint &rhs) :
            m_stack_id (rhs.m_stack_id),
            m_data_sp (rhs.m_data_sp)
        {
        }
        
        const StackID &
        GetStackID()
        {
            return m_stack_id;
        }

        void
        SetStackID (const StackID &stack_id)
        {
            m_stack_id = stack_id;
        }

        lldb::DataBufferSP &
        GetData()
        {
            return m_data_sp;
        }

        const lldb::DataBufferSP &
        GetData() const
        {
            return m_data_sp;
        }

    protected:
        StackID m_stack_id;
        lldb::DataBufferSP m_data_sp;
    };

    struct ThreadStateCheckpoint
    {
        uint32_t           orig_stop_id;  // Dunno if I need this yet but it is an interesting bit of data.
        lldb::StopInfoSP   stop_info_sp;  // You have to restore the stop info or you might continue with the wrong signals.
        RegisterCheckpoint register_backup;  // You need to restore the registers, of course...
        uint32_t           current_inlined_depth;
        lldb::addr_t       current_inlined_pc;
    };

    static void
    SettingsInitialize ();

    static void
    SettingsTerminate ();

    static const ThreadPropertiesSP &
    GetGlobalProperties();

    Thread (Process &process, lldb::tid_t tid);
    virtual ~Thread();

    lldb::ProcessSP
    GetProcess() const
    {
        return m_process_wp.lock(); 
    }

    int
    GetResumeSignal () const
    {
        return m_resume_signal;
    }

    void
    SetResumeSignal (int signal)
    {
        m_resume_signal = signal;
    }

    lldb::StateType
    GetState() const;

    void
    SetState (lldb::StateType state);

    lldb::StateType
    GetResumeState () const
    {
        return m_resume_state;
    }

    void
    SetResumeState (lldb::StateType state)
    {
        m_resume_state = state;
    }

    // This function is called on all the threads before "WillResume" in case
    // a thread needs to change its state before the ThreadList polls all the
    // threads to figure out which ones actually will get to run and how.
    void
    SetupForResume ();
    
    // Override this to do platform specific tasks before resume, but always
    // call the Thread::WillResume at the end of your work.

    virtual bool
    WillResume (lldb::StateType resume_state);

    // This clears generic thread state after a resume.  If you subclass this,
    // be sure to call it.
    virtual void
    DidResume ();

    virtual void
    RefreshStateAfterStop() = 0;

    void
    WillStop ();

    bool
    ShouldStop (Event *event_ptr);

    Vote
    ShouldReportStop (Event *event_ptr);

    Vote
    ShouldReportRun (Event *event_ptr);
    
    void
    Flush ();

    // Return whether this thread matches the specification in ThreadSpec.  This is a virtual
    // method because at some point we may extend the thread spec with a platform specific
    // dictionary of attributes, which then only the platform specific Thread implementation
    // would know how to match.  For now, this just calls through to the ThreadSpec's 
    // ThreadPassesBasicTests method.
    virtual bool
    MatchesSpec (const ThreadSpec *spec);

    lldb::StopInfoSP
    GetStopInfo ();

    lldb::StopReason
    GetStopReason();

    // This sets the stop reason to a "blank" stop reason, so you can call functions on the thread
    // without having the called function run with whatever stop reason you stopped with.
    void
    SetStopInfoToNothing();
    
    bool
    ThreadStoppedForAReason ();

    static const char *
    RunModeAsCString (lldb::RunMode mode);

    static const char *
    StopReasonAsCString (lldb::StopReason reason);

    virtual const char *
    GetInfo ()
    {
        return NULL;
    }

    virtual const char *
    GetName ()
    {
        return NULL;
    }

    virtual const char *
    GetQueueName ()
    {
        return NULL;
    }

    virtual uint32_t
    GetStackFrameCount()
    {
        return GetStackFrameList()->GetNumFrames();
    }

    virtual lldb::StackFrameSP
    GetStackFrameAtIndex (uint32_t idx)
    {
        return GetStackFrameList()->GetFrameAtIndex(idx);
    }
    
    virtual lldb::StackFrameSP
    GetFrameWithConcreteFrameIndex (uint32_t unwind_idx);
    
    bool
    DecrementCurrentInlinedDepth()
    {
        return GetStackFrameList()->DecrementCurrentInlinedDepth();
    }
    
    uint32_t
    GetCurrentInlinedDepth()
    {
        return GetStackFrameList()->GetCurrentInlinedDepth();
    }
    
    Error
    ReturnFromFrameWithIndex (uint32_t frame_idx, lldb::ValueObjectSP return_value_sp, bool broadcast = false);
    
    Error
    ReturnFromFrame (lldb::StackFrameSP frame_sp, lldb::ValueObjectSP return_value_sp, bool broadcast = false);
    
    virtual lldb::StackFrameSP
    GetFrameWithStackID (const StackID &stack_id)
    {
        return GetStackFrameList()->GetFrameWithStackID (stack_id);
    }

    uint32_t
    GetSelectedFrameIndex ()
    {
        return GetStackFrameList()->GetSelectedFrameIndex();
    }

    lldb::StackFrameSP
    GetSelectedFrame ()
    {
        lldb::StackFrameListSP stack_frame_list_sp(GetStackFrameList());
        return stack_frame_list_sp->GetFrameAtIndex (stack_frame_list_sp->GetSelectedFrameIndex());
    }

    uint32_t
    SetSelectedFrame (lldb_private::StackFrame *frame, bool broadcast = false);

    bool
    SetSelectedFrameByIndex (uint32_t frame_idx, bool broadcast = false);

    void
    SetDefaultFileAndLineToSelectedFrame()
    {
        GetStackFrameList()->SetDefaultFileAndLineToSelectedFrame();
    }

    virtual lldb::RegisterContextSP
    GetRegisterContext () = 0;

    virtual lldb::RegisterContextSP
    CreateRegisterContextForFrame (StackFrame *frame) = 0;
    
    virtual void
    ClearStackFrames ();

    void
    DumpUsingSettingsFormat (Stream &strm, uint32_t frame_idx);

    //------------------------------------------------------------------
    // Thread Plan Providers:
    // This section provides the basic thread plans that the Process control
    // machinery uses to run the target.  ThreadPlan.h provides more details on
    // how this mechanism works.
    // The thread provides accessors to a set of plans that perform basic operations.
    // The idea is that particular Platform plugins can override these methods to
    // provide the implementation of these basic operations appropriate to their
    // environment.
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    /// Queues the base plan for a thread.
    /// The version returned by Process does some things that are useful,
    /// like handle breakpoints and signals, so if you return a plugin specific
    /// one you probably want to call through to the Process one for anything
    /// your plugin doesn't explicitly handle.
    ///
    /// @param[in] abort_other_plans
    ///    \b true if we discard the currently queued plans and replace them with this one.
    ///    Otherwise this plan will go on the end of the plan stack.
    ///
    /// @return
    ///     A pointer to the newly queued thread plan, or NULL if the plan could not be queued.
    //------------------------------------------------------------------
    virtual ThreadPlan *
    QueueFundamentalPlan (bool abort_other_plans);

    //------------------------------------------------------------------
    /// Queues the plan used to step over a breakpoint at the current PC of \a thread.
    /// The default version returned by Process handles trap based breakpoints, and
    /// will disable the breakpoint, single step over it, then re-enable it.
    ///
    /// @param[in] abort_other_plans
    ///    \b true if we discard the currently queued plans and replace them with this one.
    ///    Otherwise this plan will go on the end of the plan stack.
    ///
    /// @return
    ///     A pointer to the newly queued thread plan, or NULL if the plan could not be queued.
    //------------------------------------------------------------------
    virtual ThreadPlan *
    QueueThreadPlanForStepOverBreakpointPlan (bool abort_other_plans);

    //------------------------------------------------------------------
    /// Queues the plan used to step one instruction from the current PC of \a thread.
    ///
    /// @param[in] step_over
    ///    \b true if we step over calls to functions, false if we step in.
    ///
    /// @param[in] abort_other_plans
    ///    \b true if we discard the currently queued plans and replace them with this one.
    ///    Otherwise this plan will go on the end of the plan stack.
    ///
    /// @param[in] stop_other_threads
    ///    \b true if we will stop other threads while we single step this one.
    ///
    /// @return
    ///     A pointer to the newly queued thread plan, or NULL if the plan could not be queued.
    //------------------------------------------------------------------
    virtual ThreadPlan *
    QueueThreadPlanForStepSingleInstruction (bool step_over,
                                             bool abort_other_plans,
                                             bool stop_other_threads);

    //------------------------------------------------------------------
    /// Queues the plan used to step through an address range, stepping into or over
    /// function calls depending on the value of StepType.
    ///
    /// @param[in] abort_other_plans
    ///    \b true if we discard the currently queued plans and replace them with this one.
    ///    Otherwise this plan will go on the end of the plan stack.
    ///
    /// @param[in] type
    ///    Type of step to do, only eStepTypeInto and eStepTypeOver are supported by this plan.
    ///
    /// @param[in] range
    ///    The address range to step through.
    ///
    /// @param[in] addr_context
    ///    When dealing with stepping through inlined functions the current PC is not enough information to know
    ///    what "step" means.  For instance a series of nested inline functions might start at the same address.
    //     The \a addr_context provides the current symbol context the step
    ///    is supposed to be out of.
    //   FIXME: Currently unused.
    ///
    /// @param[in] stop_other_threads
    ///    \b true if we will stop other threads while we single step this one.
    ///
    /// @return
    ///     A pointer to the newly queued thread plan, or NULL if the plan could not be queued.
    //------------------------------------------------------------------
    virtual ThreadPlan *
    QueueThreadPlanForStepRange (bool abort_other_plans,
                                 StepType type,
                                 const AddressRange &range,
                                 const SymbolContext &addr_context,
                                 lldb::RunMode stop_other_threads,
                                 bool avoid_code_without_debug_info);

    //------------------------------------------------------------------
    /// Queue the plan used to step out of the function at the current PC of
    /// \a thread.
    ///
    /// @param[in] abort_other_plans
    ///    \b true if we discard the currently queued plans and replace them with this one.
    ///    Otherwise this plan will go on the end of the plan stack.
    ///
    /// @param[in] addr_context
    ///    When dealing with stepping through inlined functions the current PC is not enough information to know
    ///    what "step" means.  For instance a series of nested inline functions might start at the same address.
    //     The \a addr_context provides the current symbol context the step
    ///    is supposed to be out of.
    //   FIXME: Currently unused.
    ///
    /// @param[in] first_insn
    ///     \b true if this is the first instruction of a function.
    ///
    /// @param[in] stop_other_threads
    ///    \b true if we will stop other threads while we single step this one.
    ///
    /// @param[in] stop_vote
    /// @param[in] run_vote
    ///    See standard meanings for the stop & run votes in ThreadPlan.h.
    ///
    /// @return
    ///     A pointer to the newly queued thread plan, or NULL if the plan could not be queued.
    //------------------------------------------------------------------
    virtual ThreadPlan *
    QueueThreadPlanForStepOut (bool abort_other_plans,
                               SymbolContext *addr_context,
                               bool first_insn,
                               bool stop_other_threads,
                               Vote stop_vote, // = eVoteYes,
                               Vote run_vote, // = eVoteNoOpinion);
                               uint32_t frame_idx);

    //------------------------------------------------------------------
    /// Gets the plan used to step through the code that steps from a function
    /// call site at the current PC into the actual function call.
    ///
    ///
    /// @param[in] return_stack_id
    ///    The stack id that we will return to (by setting backstop breakpoints on the return
    ///    address to that frame) if we fail to step through.
    ///
    /// @param[in] abort_other_plans
    ///    \b true if we discard the currently queued plans and replace them with this one.
    ///    Otherwise this plan will go on the end of the plan stack.
    ///
    /// @param[in] stop_other_threads
    ///    \b true if we will stop other threads while we single step this one.
    ///
    /// @return
    ///     A pointer to the newly queued thread plan, or NULL if the plan could not be queued.
    //------------------------------------------------------------------
    virtual ThreadPlan *
    QueueThreadPlanForStepThrough (StackID &return_stack_id,
                                   bool abort_other_plans,
                                   bool stop_other_threads);

    //------------------------------------------------------------------
    /// Gets the plan used to continue from the current PC.
    /// This is a simple plan, mostly useful as a backstop when you are continuing
    /// for some particular purpose.
    ///
    /// @param[in] abort_other_plans
    ///    \b true if we discard the currently queued plans and replace them with this one.
    ///    Otherwise this plan will go on the end of the plan stack.
    ///
    /// @param[in] target_addr
    ///    The address to which we're running.
    ///
    /// @param[in] stop_other_threads
    ///    \b true if we will stop other threads while we single step this one.
    ///
    /// @return
    ///     A pointer to the newly queued thread plan, or NULL if the plan could not be queued.
    //------------------------------------------------------------------
    virtual ThreadPlan *
    QueueThreadPlanForRunToAddress (bool abort_other_plans,
                                    Address &target_addr,
                                    bool stop_other_threads);

    virtual ThreadPlan *
    QueueThreadPlanForStepUntil (bool abort_other_plans,
                                 lldb::addr_t *address_list,
                                 size_t num_addresses,
                                 bool stop_others,
                                 uint32_t frame_idx);

    virtual ThreadPlan *
    QueueThreadPlanForCallFunction (bool abort_other_plans,
                                    Address& function,
                                    lldb::addr_t arg,
                                    bool stop_other_threads,
                                    bool discard_on_error = false);
                                            
    //------------------------------------------------------------------
    // Thread Plan accessors:
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    /// Gets the plan which will execute next on the plan stack.
    ///
    /// @return
    ///     A pointer to the next executed plan.
    //------------------------------------------------------------------
    ThreadPlan *
    GetCurrentPlan ();

private:
    bool
    PlanIsBasePlan (ThreadPlan *plan_ptr);
    
    void
    BroadcastSelectedFrameChange(StackID &new_frame_id);
    
public:

    //------------------------------------------------------------------
    /// Gets the outer-most plan that was popped off the plan stack in the
    /// most recent stop.  Useful for printing the stop reason accurately.
    ///
    /// @return
    ///     A pointer to the last completed plan.
    //------------------------------------------------------------------
    lldb::ThreadPlanSP
    GetCompletedPlan ();

    //------------------------------------------------------------------
    /// Gets the outer-most return value from the completed plans
    ///
    /// @return
    ///     A ValueObjectSP, either empty if there is no return value,
    ///     or containing the return value.
    //------------------------------------------------------------------
    lldb::ValueObjectSP
    GetReturnValueObject ();

    //------------------------------------------------------------------
    ///  Checks whether the given plan is in the completed plans for this
    ///  stop.
    ///
    /// @param[in] plan
    ///     Pointer to the plan you're checking.
    ///
    /// @return
    ///     Returns true if the input plan is in the completed plan stack,
    ///     false otherwise.
    //------------------------------------------------------------------
    bool
    IsThreadPlanDone (ThreadPlan *plan);

    //------------------------------------------------------------------
    ///  Checks whether the given plan is in the discarded plans for this
    ///  stop.
    ///
    /// @param[in] plan
    ///     Pointer to the plan you're checking.
    ///
    /// @return
    ///     Returns true if the input plan is in the discarded plan stack,
    ///     false otherwise.
    //------------------------------------------------------------------
    bool
    WasThreadPlanDiscarded (ThreadPlan *plan);

    //------------------------------------------------------------------
    /// Queues a generic thread plan.
    ///
    /// @param[in] plan_sp
    ///    The plan to queue.
    ///
    /// @param[in] abort_other_plans
    ///    \b true if we discard the currently queued plans and replace them with this one.
    ///    Otherwise this plan will go on the end of the plan stack.
    ///
    /// @return
    ///     A pointer to the last completed plan.
    //------------------------------------------------------------------
    void
    QueueThreadPlan (lldb::ThreadPlanSP &plan_sp, bool abort_other_plans);


    //------------------------------------------------------------------
    /// Discards the plans queued on the plan stack of the current thread.  This is
    /// arbitrated by the "Master" ThreadPlans, using the "OkayToDiscard" call.
    //  But if \a force is true, all thread plans are discarded.
    //------------------------------------------------------------------
    void
    DiscardThreadPlans (bool force);

    //------------------------------------------------------------------
    /// Discards the plans queued on the plan stack of the current thread up to and
    /// including up_to_plan_sp.
    //
    // @param[in] up_to_plan_sp
    //   Discard all plans up to and including this one.
    //------------------------------------------------------------------
    void
    DiscardThreadPlansUpToPlan (lldb::ThreadPlanSP &up_to_plan_sp);

    void
    DiscardThreadPlansUpToPlan (ThreadPlan *up_to_plan_ptr);
    
    //------------------------------------------------------------------
    /// Prints the current plan stack.
    ///
    /// @param[in] s
    ///    The stream to which to dump the plan stack info.
    ///
    //------------------------------------------------------------------
    void
    DumpThreadPlans (Stream *s) const;
    
    virtual bool
    CheckpointThreadState (ThreadStateCheckpoint &saved_state);
    
    virtual bool
    RestoreThreadStateFromCheckpoint (ThreadStateCheckpoint &saved_state);
    
    void
    EnableTracer (bool value, bool single_step);
    
    void
    SetTracer (lldb::ThreadPlanTracerSP &tracer_sp);
    
    // Get the thread index ID. The index ID that is guaranteed to not be
    // re-used by a process. They start at 1 and increase with each new thread.
    // This allows easy command line access by a unique ID that is easier to
    // type than the actual system thread ID.
    uint32_t
    GetIndexID () const;
    
    //------------------------------------------------------------------
    // lldb::ExecutionContextScope pure virtual functions
    //------------------------------------------------------------------
    virtual lldb::TargetSP
    CalculateTarget ();
    
    virtual lldb::ProcessSP
    CalculateProcess ();
    
    virtual lldb::ThreadSP
    CalculateThread ();
    
    virtual lldb::StackFrameSP
    CalculateStackFrame ();

    virtual void
    CalculateExecutionContext (ExecutionContext &exe_ctx);
    
    lldb::StackFrameSP
    GetStackFrameSPForStackFramePtr (StackFrame *stack_frame_ptr);
    
    size_t
    GetStatus (Stream &strm, 
               uint32_t start_frame, 
               uint32_t num_frames,
               uint32_t num_frames_with_source);

    size_t
    GetStackFrameStatus (Stream& strm,
                         uint32_t first_frame,
                         uint32_t num_frames,
                         bool show_frame_info,
                         uint32_t num_frames_with_source);

    // We need a way to verify that even though we have a thread in a shared
    // pointer that the object itself is still valid. Currently this won't be
    // the case if DestroyThread() was called. DestroyThread is called when
    // a thread has been removed from the Process' thread list.
    bool
    IsValid () const
    {
        return !m_destroy_called;
    }

    // When you implement this method, make sure you don't overwrite the m_actual_stop_info if it claims to be
    // valid.  The stop info may be a "checkpointed and restored" stop info, so if it is still around it is right
    // even if you have not calculated this yourself, or if it disagrees with what you might have calculated.
    virtual lldb::StopInfoSP
    GetPrivateStopReason () = 0;

    //----------------------------------------------------------------------
    // Gets the temporary resume state for a thread.
    //
    // This value gets set in each thread by complex debugger logic in
    // Thread::WillResume() and an appropriate thread resume state will get
    // set in each thread every time the process is resumed prior to calling
    // Process::DoResume(). The lldb_private::Process subclass should adhere
    // to the thread resume state request which will be one of:
    //
    //  eStateRunning   - thread will resume when process is resumed
    //  eStateStepping  - thread should step 1 instruction and stop when process
    //                    is resumed
    //  eStateSuspended - thread should not execute any instructions when
    //                    process is resumed
    //----------------------------------------------------------------------
    lldb::StateType
    GetTemporaryResumeState() const
    {
        return m_temporary_resume_state;
    }

protected:

    friend class ThreadPlan;
    friend class ThreadList;
    friend class StackFrameList;
    friend class StackFrame;
    
    // This is necessary to make sure thread assets get destroyed while the thread is still in good shape
    // to call virtual thread methods.  This must be called by classes that derive from Thread in their destructor.
    virtual void DestroyThread ();

    void
    PushPlan (lldb::ThreadPlanSP &plan_sp);

    void
    PopPlan ();

    void
    DiscardPlan ();

    ThreadPlan *GetPreviousPlan (ThreadPlan *plan);

    typedef std::vector<lldb::ThreadPlanSP> plan_stack;

    void
    SetStopInfo (const lldb::StopInfoSP &stop_info_sp);

    virtual bool
    SaveFrameZeroState (RegisterCheckpoint &checkpoint);

    virtual bool
    RestoreSaveFrameZero (const RegisterCheckpoint &checkpoint);
    
    // register_data_sp must be a DataSP passed to ReadAllRegisterValues.
    bool
    ResetFrameZeroRegisters (lldb::DataBufferSP register_data_sp);

    virtual lldb_private::Unwind *
    GetUnwinder ();

    // Check to see whether the thread is still at the last breakpoint hit that stopped it.
    virtual const bool
    IsStillAtLastBreakpointHit();

    lldb::StackFrameListSP
    GetStackFrameList ();
    
    struct ThreadState
    {
        uint32_t           orig_stop_id;
        lldb::StopInfoSP   stop_info_sp;
        RegisterCheckpoint register_backup;
    };

    //------------------------------------------------------------------
    // Classes that inherit from Process can see and modify these
    //------------------------------------------------------------------
    lldb::ProcessWP     m_process_wp;           ///< The process that owns this thread.
    lldb::StopInfoSP    m_actual_stop_info_sp;  ///< The private stop reason for this thread
    const uint32_t      m_index_id;             ///< A unique 1 based index assigned to each thread for easy UI/command line access.
    lldb::RegisterContextSP m_reg_context_sp;   ///< The register context for this thread's current register state.
    lldb::StateType     m_state;                ///< The state of our process.
    mutable Mutex       m_state_mutex;          ///< Multithreaded protection for m_state.
    plan_stack          m_plan_stack;           ///< The stack of plans this thread is executing.
    plan_stack          m_completed_plan_stack; ///< Plans that have been completed by this stop.  They get deleted when the thread resumes.
    plan_stack          m_discarded_plan_stack; ///< Plans that have been discarded by this stop.  They get deleted when the thread resumes.
    mutable Mutex       m_frame_mutex;          ///< Multithreaded protection for m_state.
    lldb::StackFrameListSP m_curr_frames_sp;    ///< The stack frames that get lazily populated after a thread stops.
    lldb::StackFrameListSP m_prev_frames_sp;    ///< The previous stack frames from the last time this thread stopped.
    int                 m_resume_signal;        ///< The signal that should be used when continuing this thread.
    lldb::StateType     m_resume_state;         ///< This state is used to force a thread to be suspended from outside the ThreadPlan logic.
    lldb::StateType     m_temporary_resume_state; ///< This state records what the thread was told to do by the thread plan logic for the current resume.
                                                  /// It gets set in Thread::WillResume.
    std::auto_ptr<lldb_private::Unwind> m_unwinder_ap;
    bool                m_destroy_called;       // This is used internally to make sure derived Thread classes call DestroyThread.
    uint32_t m_thread_stop_reason_stop_id;      // This is the stop id for which the StopInfo is valid.  Can use this so you know that
                                                // the thread's m_actual_stop_info_sp is current and you don't have to fetch it again

private:
    //------------------------------------------------------------------
    // For Thread only
    //------------------------------------------------------------------

    DISALLOW_COPY_AND_ASSIGN (Thread);

};

} // namespace lldb_private

#endif  // liblldb_Thread_h_
