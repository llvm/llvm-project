//===-- SWIG Interface for SBBreakpoint -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Represents a logical breakpoint and its associated settings.

For example (from test/functionalities/breakpoint/breakpoint_ignore_count/
TestBreakpointIgnoreCount.py),

    def breakpoint_ignore_count_python(self):
        '''Use Python APIs to set breakpoint ignore count.'''
        exe = os.path.join(os.getcwd(), 'a.out')

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint on main.c by name 'c'.
        breakpoint = target.BreakpointCreateByName('c', 'a.out')
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        # Get the breakpoint location from breakpoint after we verified that,
        # indeed, it has one location.
        location = breakpoint.GetLocationAtIndex(0)
        self.assertTrue(location and
                        location.IsEnabled(),
                        VALID_BREAKPOINT_LOCATION)

        # Set the ignore count on the breakpoint location.
        location.SetIgnoreCount(2)
        self.assertTrue(location.GetIgnoreCount() == 2,
                        'SetIgnoreCount() works correctly')

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(None, None, os.getcwd())
        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame#0 should be on main.c:37, frame#1 should be on main.c:25, and
        # frame#2 should be on main.c:48.
        #lldbutil.print_stacktraces(process)
        from lldbutil import get_stopped_thread
        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread != None, 'There should be a thread stopped due to breakpoint')
        frame0 = thread.GetFrameAtIndex(0)
        frame1 = thread.GetFrameAtIndex(1)
        frame2 = thread.GetFrameAtIndex(2)
        self.assertTrue(frame0.GetLineEntry().GetLine() == self.line1 and
                        frame1.GetLineEntry().GetLine() == self.line3 and
                        frame2.GetLineEntry().GetLine() == self.line4,
                        STOPPED_DUE_TO_BREAKPOINT_IGNORE_COUNT)

        # The hit count for the breakpoint should be 3.
        self.assertTrue(breakpoint.GetHitCount() == 3)

        process.Continue()

SBBreakpoint supports breakpoint location iteration, for example,

    for bl in breakpoint:
        print 'breakpoint location load addr: %s' % hex(bl.GetLoadAddress())
        print 'breakpoint location condition: %s' % hex(bl.GetCondition())

and rich comparion methods which allow the API program to use,

    if aBreakpoint == bBreakpoint:
        ...

to compare two breakpoints for equality."
) SBBreakpoint;
class SBBreakpoint
{
public:

    typedef bool (*BreakpointHitCallback) (void *baton, 
                                           SBProcess &process,
                                           SBThread &thread, 
                                           lldb::SBBreakpointLocation &location);

    SBBreakpoint ();

    SBBreakpoint (const lldb::SBBreakpoint& rhs);

    ~SBBreakpoint();

    break_id_t
    GetID () const;

    bool
    IsValid() const;

    void
    ClearAllBreakpointSites ();

    lldb::SBBreakpointLocation
    FindLocationByAddress (lldb::addr_t vm_addr);

    lldb::break_id_t
    FindLocationIDByAddress (lldb::addr_t vm_addr);

    lldb::SBBreakpointLocation
    FindLocationByID (lldb::break_id_t bp_loc_id);

    lldb::SBBreakpointLocation
    GetLocationAtIndex (uint32_t index);

    void
    SetEnabled (bool enable);

    bool
    IsEnabled ();
    
    void
    SetOneShot (bool one_shot);

    bool
    IsOneShot ();
    
    bool
    IsInternal ();

    uint32_t
    GetHitCount () const;

    void
    SetIgnoreCount (uint32_t count);

    uint32_t
    GetIgnoreCount () const;
    
    %feature("docstring", "
    //--------------------------------------------------------------------------
    /// The breakpoint stops only if the condition expression evaluates to true.
    //--------------------------------------------------------------------------
    ") SetCondition;
    void 
    SetCondition (const char *condition);
    
    %feature("docstring", "
    //------------------------------------------------------------------
    /// Get the condition expression for the breakpoint.
    //------------------------------------------------------------------
    ") GetCondition;
    const char *
    GetCondition ();

    void
    SetThreadID (lldb::tid_t sb_thread_id);

    lldb::tid_t
    GetThreadID ();
    
    void
    SetThreadIndex (uint32_t index);
    
    uint32_t
    GetThreadIndex() const;
    
    void
    SetThreadName (const char *thread_name);
    
    const char *
    GetThreadName () const;
    
    void 
    SetQueueName (const char *queue_name);
    
    const char *
    GetQueueName () const;

    void
    SetCallback (BreakpointHitCallback callback, void *baton);

    size_t
    GetNumResolvedLocations() const;

    size_t
    GetNumLocations() const;

    bool
    GetDescription (lldb::SBStream &description);

    static bool
    EventIsBreakpointEvent (const lldb::SBEvent &event);
    
    static lldb::BreakpointEventType
    GetBreakpointEventTypeFromEvent (const lldb::SBEvent& event);

    static lldb::SBBreakpoint
    GetBreakpointFromEvent (const lldb::SBEvent& event);
    
    static lldb::SBBreakpointLocation
    GetBreakpointLocationAtIndexFromEvent (const lldb::SBEvent& event, uint32_t loc_idx);
    
    static uint32_t
    GetNumBreakpointLocationsFromEvent (const lldb::SBEvent &event_sp);
};

} // namespace lldb
