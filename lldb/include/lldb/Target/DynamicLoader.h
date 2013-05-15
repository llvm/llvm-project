//===-- DynamicLoader.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_DynamicLoader_h_
#define liblldb_DynamicLoader_h_

// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/PluginInterface.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class DynamicLoader DynamicLoader.h "lldb/Target/DynamicLoader.h"
/// @brief A plug-in interface definition class for dynamic loaders.
///
/// Dynamic loader plug-ins track image (shared library) loading and
/// unloading. The class is initialized given a live process that is
/// halted at its entry point or just after attaching.
///
/// Dynamic loader plug-ins can track the process by registering
/// callbacks using the:
/// Process::RegisterNotificationCallbacks (const Notifications&)
/// function.
///
/// Breakpoints can also be set in the process which can register
/// functions that get called using:
/// Process::BreakpointSetCallback (lldb::user_id_t, BreakpointHitCallback, void *).
/// These breakpoint callbacks return a boolean value that indicates if
/// the process should continue or halt and should return the global
/// setting for this using:
/// DynamicLoader::StopWhenImagesChange() const.
//----------------------------------------------------------------------
class DynamicLoader :
    public PluginInterface
{
public:
    //------------------------------------------------------------------
    /// Find a dynamic loader plugin for a given process.
    ///
    /// Scans the installed DynamicLoader plug-ins and tries to find
    /// an instance that can be used to track image changes in \a
    /// process.
    ///
    /// @param[in] process
    ///     The process for which to try and locate a dynamic loader
    ///     plug-in instance.
    ///
    /// @param[in] plugin_name
    ///     An optional name of a specific dynamic loader plug-in that
    ///     should be used. If NULL, pick the best plug-in.
    //------------------------------------------------------------------
    static DynamicLoader*
    FindPlugin (Process *process, const char *plugin_name);

    //------------------------------------------------------------------
    /// Construct with a process.
    //------------------------------------------------------------------
    DynamicLoader (Process *process);

    //------------------------------------------------------------------
    /// Destructor.
    ///
    /// The destructor is virtual since this class is designed to be
    /// inherited from by the plug-in instance.
    //------------------------------------------------------------------
    virtual
    ~DynamicLoader ();

    //------------------------------------------------------------------
    /// Called after attaching a process.
    ///
    /// Allow DynamicLoader plug-ins to execute some code after
    /// attaching to a process.
    //------------------------------------------------------------------
    virtual void
    DidAttach () = 0;

    //------------------------------------------------------------------
    /// Called after launching a process.
    ///
    /// Allow DynamicLoader plug-ins to execute some code after
    /// the process has stopped for the first time on launch.
    //------------------------------------------------------------------
    virtual void
    DidLaunch () = 0;
    
    
    //------------------------------------------------------------------
    /// Helper function that can be used to detect when a process has
    /// called exec and is now a new and different process. This can
    /// be called when necessary to try and detect the exec. The process
    /// might be able to answer this question, but sometimes it might
    /// not be able and the dynamic loader often knows what the program
    /// entry point is. So the process and the dynamic loader can work
    /// together to detect this.
    //------------------------------------------------------------------
    virtual bool
    ProcessDidExec ()
    {
        return false;
    }
    //------------------------------------------------------------------
    /// Get whether the process should stop when images change.
    ///
    /// When images (executables and shared libraries) get loaded or
    /// unloaded, often debug sessions will want to try and resolve or
    /// unresolve breakpoints that are set in these images. Any
    /// breakpoints set by DynamicLoader plug-in instances should
    /// return this value to ensure consistent debug session behaviour.
    ///
    /// @return
    ///     Returns \b true if the process should stop when images
    ///     change, \b false if the process should resume.
    //------------------------------------------------------------------
    bool
    GetStopWhenImagesChange () const;

    //------------------------------------------------------------------
    /// Set whether the process should stop when images change.
    ///
    /// When images (executables and shared libraries) get loaded or
    /// unloaded, often debug sessions will want to try and resolve or
    /// unresolve breakpoints that are set in these images. The default
    /// is set so that the process stops when images change, but this
    /// can be overridden using this function callback.
    ///
    /// @param[in] stop
    ///     Boolean value that indicates whether the process should stop
    ///     when images change.
    //------------------------------------------------------------------
    void
    SetStopWhenImagesChange (bool stop);

    //------------------------------------------------------------------
    /// Provides a plan to step through the dynamic loader trampoline
    /// for the current state of \a thread.
    ///
    ///
    /// @param[in] stop_others
    ///     Whether the plan should be set to stop other threads.
    ///
    /// @return
    ///    A pointer to the plan (caller owned) or NULL if we are not at such
    ///    a trampoline.
    //------------------------------------------------------------------
    virtual lldb::ThreadPlanSP
    GetStepThroughTrampolinePlan (Thread &thread, bool stop_others) = 0;


    //------------------------------------------------------------------
    /// Some dynamic loaders provide features where there are a group of symbols "equivalent to"
    /// a given symbol one of which will be chosen when the symbol is bound.  If you want to
    /// set a breakpoint on one of these symbols, you really need to set it on all the
    /// equivalent symbols.
    ///
    ///
    /// @param[in] original_symbol
    ///     The symbol for which we are finding equivalences.
    ///
    /// @param[in] module_list
    ///     The set of modules in which to search.
    ///
    /// @param[out] equivalent_symbols
    ///     The equivalent symbol list - any equivalent symbols found are appended to this list.
    ///
    /// @return
    ///    Number of equivalent symbols found.
    //------------------------------------------------------------------
    virtual size_t
    FindEquivalentSymbols (Symbol *original_symbol, ModuleList &module_list, SymbolContextList &equivalent_symbols)
    {
        return 0;
    }

    //------------------------------------------------------------------
    /// Ask if it is ok to try and load or unload an shared library 
    /// (image).
    ///
    /// The dynamic loader often knows when it would be ok to try and
    /// load or unload a shared library. This function call allows the
    /// dynamic loader plug-ins to check any current dyld state to make
    /// sure it is an ok time to load a shared library.
    ///
    /// @return
    ///     \b true if it is currently ok to try and load a shared 
    ///     library into the process, \b false otherwise.
    //------------------------------------------------------------------
    virtual Error
    CanLoadImage () = 0;

    //------------------------------------------------------------------
    /// Ask if the eh_frame information for the given SymbolContext should
    /// be relied on even when it's the first frame in a stack unwind.
    /// 
    /// The CFI instructions from the eh_frame section are normally only
    /// valid at call sites -- places where a program could throw an 
    /// exception and need to unwind out.  But some Modules may be known 
    /// to the system as having reliable eh_frame information at all call 
    /// sites.  This would be the case if the Module's contents are largely
    /// hand-written assembly with hand-written eh_frame information.
    /// Normally when unwinding from a function at the beginning of a stack
    /// unwind lldb will examine the assembly instructions to understand
    /// how the stack frame is set up and where saved registers are stored.
    /// But with hand-written assembly this is not reliable enough -- we need
    /// to consult those function's hand-written eh_frame information.
    ///
    /// @return
    ///     \b True if the symbol context should use eh_frame instructions
    ///     unconditionally when unwinding from this frame.  Else \b false,
    ///     the normal lldb unwind behavior of only using eh_frame when the
    ///     function appears in the middle of the stack.
    //------------------------------------------------------------------
    virtual bool
    AlwaysRelyOnEHUnwindInfo (SymbolContext &sym_ctx)
    {
       return false;
    }

protected:
    //------------------------------------------------------------------
    // Member variables.
    //------------------------------------------------------------------
    Process* m_process; ///< The process that this dynamic loader plug-in is tracking.
private:
    DISALLOW_COPY_AND_ASSIGN (DynamicLoader);

};

} // namespace lldb_private

#endif  // liblldb_DynamicLoader_h_
