//===-- BreakpointOptions.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_BreakpointOptions_h_
#define liblldb_BreakpointOptions_h_

// C Includes
// C++ Includes
#include <memory>
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Core/Baton.h"
#include "lldb/Core/StringList.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class BreakpointOptions BreakpointOptions.h "lldb/Breakpoint/BreakpointOptions.h"
/// @brief Class that manages the options on a breakpoint or breakpoint location.
//----------------------------------------------------------------------

class BreakpointOptions
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    //------------------------------------------------------------------
    /// Default constructor.  The breakpoint is enabled, and has no condition,
    /// callback, ignore count, etc...
    //------------------------------------------------------------------
    BreakpointOptions();
    BreakpointOptions(const BreakpointOptions& rhs);

    static BreakpointOptions *
    CopyOptionsNoCallback (BreakpointOptions &rhs);
    //------------------------------------------------------------------
    /// This constructor allows you to specify all the breakpoint options.
    ///
    /// @param[in] condition
    ///    The expression which if it evaluates to \b true if we are to stop
    ///
    /// @param[in] callback
    ///    This is the plugin for some code that gets run, returns \b true if we are to stop.
    ///
    /// @param[in] baton
    ///    Client data that will get passed to the callback.
    ///
    /// @param[in] enabled
    ///    Is this breakpoint enabled.
    ///
    /// @param[in] ignore
    ///    How many breakpoint hits we should ignore before stopping.
    ///
    /// @param[in] thread_id
    ///    Only stop if \a thread_id hits the breakpoint.
    //------------------------------------------------------------------
    BreakpointOptions(void *condition,
                      BreakpointHitCallback callback,
                      void *baton,
                      bool enabled = true,
                      int32_t ignore = 0,
                      lldb::tid_t thread_id = LLDB_INVALID_THREAD_ID,
                      bool one_shot = false);

    virtual ~BreakpointOptions();

    //------------------------------------------------------------------
    // Operators
    //------------------------------------------------------------------
    const BreakpointOptions&
    operator=(const BreakpointOptions& rhs);

    //------------------------------------------------------------------
    // Callbacks
    //
    // Breakpoint callbacks come in two forms, synchronous and asynchronous.  Synchronous callbacks will get
    // run before any of the thread plans are consulted, and if they return false the target will continue
    // "under the radar" of the thread plans.  There are a couple of restrictions to synchronous callbacks:
    // 1) They should NOT resume the target themselves.  Just return false if you want the target to restart.
    // 2) Breakpoints with synchronous callbacks can't have conditions (or rather, they can have them, but they
    //    won't do anything.  Ditto with ignore counts, etc...  You are supposed to control that all through the
    //    callback.
    // Asynchronous callbacks get run as part of the "ShouldStop" logic in the thread plan.  The logic there is:
    //   a) If the breakpoint is thread specific and not for this thread, continue w/o running the callback.
    //   b) If the ignore count says we shouldn't stop, then ditto.
    //   c) If the condition says we shouldn't stop, then ditto.
    //   d) Otherwise, the callback will get run, and if it returns true we will stop, and if false we won't.
    //  The asynchronous callback can run the target itself, but at present that should be the last action the
    //  callback does.  We will relax this condition at some point, but it will take a bit of plumbing to get
    //  that to work.
    // 
    //------------------------------------------------------------------
    
    //------------------------------------------------------------------
    /// Adds a callback to the breakpoint option set.
    ///
    /// @param[in] callback
    ///    The function to be called when the breakpoint gets hit.
    ///
    /// @param[in] baton_sp
    ///    A baton which will get passed back to the callback when it is invoked.
    ///
    /// @param[in] synchronous
    ///    Whether this is a synchronous or asynchronous callback.  See discussion above.
    //------------------------------------------------------------------
    void SetCallback (BreakpointHitCallback callback, const lldb::BatonSP &baton_sp, bool synchronous = false);
    
    
    //------------------------------------------------------------------
    /// Remove the callback from this option set.
    //------------------------------------------------------------------
    void ClearCallback ();

    // The rest of these functions are meant to be used only within the breakpoint handling mechanism.
    
    //------------------------------------------------------------------
    /// Use this function to invoke the callback for a specific stop.
    ///
    /// @param[in] context
    ///    The context in which the callback is to be invoked.  This includes the stop event, the
    ///    execution context of the stop (since you might hit the same breakpoint on multiple threads) and
    ///    whether we are currently executing synchronous or asynchronous callbacks.
    /// 
    /// @param[in] break_id
    ///    The breakpoint ID that owns this option set.
    ///
    /// @param[in] break_loc_id
    ///    The breakpoint location ID that owns this option set.
    ///
    /// @return
    ///     The callback return value.
    //------------------------------------------------------------------
    bool InvokeCallback (StoppointCallbackContext *context, lldb::user_id_t break_id, lldb::user_id_t break_loc_id);
    
    //------------------------------------------------------------------
    /// Used in InvokeCallback to tell whether it is the right time to run this kind of callback.
    ///
    /// @return
    ///     The synchronicity of our callback.
    //------------------------------------------------------------------
    bool IsCallbackSynchronous () {
        return m_callback_is_synchronous;
    }
    
    //------------------------------------------------------------------
    /// Fetch the baton from the callback.
    ///
    /// @return
    ///     The baton.
    //------------------------------------------------------------------
    Baton *GetBaton ();
    
    //------------------------------------------------------------------
    /// Fetch  a const version of the baton from the callback.
    ///
    /// @return
    ///     The baton.
    //------------------------------------------------------------------
    const Baton *GetBaton () const;
    
    //------------------------------------------------------------------
    // Condition
    //------------------------------------------------------------------
    //------------------------------------------------------------------
    /// Set the breakpoint option's condition.
    ///
    /// @param[in] condition
    ///    The condition expression to evaluate when the breakpoint is hit.
    //------------------------------------------------------------------
    void SetCondition (const char *condition);
    
    //------------------------------------------------------------------
    /// Return a pointer to the text of the condition expression.
    ///
    /// @return
    ///    A pointer to the condition expression text, or NULL if no
    //     condition has been set.
    //------------------------------------------------------------------
    const char *GetConditionText () const;
    
    //------------------------------------------------------------------
    // Enabled/Ignore Count
    //------------------------------------------------------------------

    //------------------------------------------------------------------
    /// Check the Enable/Disable state.
    /// @return
    ///     \b true if the breakpoint is enabled, \b false if disabled.
    //------------------------------------------------------------------
    bool         
    IsEnabled () const
    {
        return m_enabled;
    }

    //------------------------------------------------------------------
    /// If \a enable is \b true, enable the breakpoint, if \b false disable it.
    //------------------------------------------------------------------
    void
    SetEnabled (bool enabled)
    {
        m_enabled = enabled;
    }

    //------------------------------------------------------------------
    /// Check the One-shot state.
    /// @return
    ///     \b true if the breakpoint is one-shot, \b false otherwise.
    //------------------------------------------------------------------
    bool         
    IsOneShot () const
    {
        return m_one_shot;
    }

    //------------------------------------------------------------------
    /// If \a enable is \b true, enable the breakpoint, if \b false disable it.
    //------------------------------------------------------------------
    void
    SetOneShot (bool one_shot)
    {
        m_one_shot = one_shot;
    }

    //------------------------------------------------------------------
    /// Set the breakpoint to ignore the next \a count breakpoint hits.
    /// @param[in] count
    ///    The number of breakpoint hits to ignore.
    //------------------------------------------------------------------

    void
    SetIgnoreCount (uint32_t n)
    {
        m_ignore_count = n;
    }

    //------------------------------------------------------------------
    /// Return the current Ignore Count.
    /// @return
    ///     The number of breakpoint hits to be ignored.
    //------------------------------------------------------------------
    uint32_t
    GetIgnoreCount () const
    {
        return m_ignore_count;
    }

    //------------------------------------------------------------------
    /// Return the current thread spec for this option.  This will return NULL if the no thread
    /// specifications have been set for this Option yet.     
    /// @return
    ///     The thread specification pointer for this option, or NULL if none has
    ///     been set yet.
    //------------------------------------------------------------------
    const ThreadSpec *
    GetThreadSpecNoCreate () const;

    //------------------------------------------------------------------
    /// Returns a pointer to the ThreadSpec for this option, creating it.
    /// if it hasn't been created already.   This API is used for setting the
    /// ThreadSpec items for this option.
    //------------------------------------------------------------------
    ThreadSpec *
    GetThreadSpec ();
    
    void
    SetThreadID(lldb::tid_t thread_id);
    
    void
    GetDescription (Stream *s, lldb::DescriptionLevel level) const;
    
    //------------------------------------------------------------------
    /// Returns true if the breakpoint option has a callback set.
    //------------------------------------------------------------------
    bool
    HasCallback();

    //------------------------------------------------------------------
    /// This is the default empty callback.
    /// @return
    ///     The thread id for which the breakpoint hit will stop, 
    ///     LLDB_INVALID_THREAD_ID for all threads.
    //------------------------------------------------------------------
    static bool 
    NullCallback (void *baton, 
                  StoppointCallbackContext *context, 
                  lldb::user_id_t break_id,
                  lldb::user_id_t break_loc_id);
    
    
    struct CommandData
    {
        CommandData () :
            user_source(),
            script_source(),
            stop_on_error(true)
        {
        }

        ~CommandData ()
        {
        }
        
        StringList user_source;
        std::string script_source;
        bool stop_on_error;
    };

    class CommandBaton : public Baton
    {
    public:
        CommandBaton (CommandData *data) :
            Baton (data)
        {
        }

        virtual
        ~CommandBaton ()
        {
            delete ((CommandData *)m_data);
            m_data = NULL;
        }
        
        virtual void
        GetDescription (Stream *s, lldb::DescriptionLevel level) const;

    };

protected:
    //------------------------------------------------------------------
    // Classes that inherit from BreakpointOptions can see and modify these
    //------------------------------------------------------------------

private:
    //------------------------------------------------------------------
    // For BreakpointOptions only
    //------------------------------------------------------------------
    BreakpointHitCallback m_callback; // This is the callback function pointer
    lldb::BatonSP m_callback_baton_sp; // This is the client data for the callback
    bool m_callback_is_synchronous;
    bool m_enabled;
    bool m_one_shot;
    uint32_t m_ignore_count; // Number of times to ignore this breakpoint
    std::auto_ptr<ThreadSpec> m_thread_spec_ap; // Thread for which this breakpoint will take
    std::auto_ptr<ClangUserExpression> m_condition_ap;  // The condition to test.

};

} // namespace lldb_private

#endif  // liblldb_BreakpointOptions_h_
