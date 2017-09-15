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
        print('breakpoint location load addr: %s' % hex(bl.GetLoadAddress()))
        print('breakpoint location condition: %s' % hex(bl.GetCondition()))

and rich comparison methods which allow the API program to use,

    if aBreakpoint == bBreakpoint:
        ...

to compare two breakpoints for equality."
) SBBreakpoint;
class SBBreakpoint
{
public:

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

    void SetAutoContinue(bool auto_continue);

    bool GetAutoContinue();

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

    %feature("docstring", "
    //------------------------------------------------------------------
    /// Set the name of the script function to be called when the breakpoint is hit.
    //------------------------------------------------------------------
    ") SetScriptCallbackFunction;
    void
    SetScriptCallbackFunction (const char *callback_function_name);

    %feature("docstring", "
    //------------------------------------------------------------------
    /// Provide the body for the script function to be called when the breakpoint is hit.
    /// The body will be wrapped in a function, which be passed two arguments:
    /// 'frame' - which holds the bottom-most SBFrame of the thread that hit the breakpoint
    /// 'bpno'  - which is the SBBreakpointLocation to which the callback was attached.
    ///
    /// The error parameter is currently ignored, but will at some point hold the Python
    /// compilation diagnostics.
    /// Returns true if the body compiles successfully, false if not.
    //------------------------------------------------------------------
    ") SetScriptCallbackBody;
    SBError
    SetScriptCallbackBody (const char *script_body_text);
    
    void SetCommandLineCommands(SBStringList &commands);

    bool GetCommandLineCommands(SBStringList &commands);
    
    bool
    AddName (const char *new_name);

    void
    RemoveName (const char *name_to_remove);

    bool
    MatchesName (const char *name);
    
    void
    GetNames (SBStringList &names);
    
    size_t
    GetNumResolvedLocations() const;

    size_t
    GetNumLocations() const;

    bool
    GetDescription (lldb::SBStream &description);

    bool 
    GetDescription(lldb::SBStream &description, bool include_locations);

    bool
    operator == (const lldb::SBBreakpoint& rhs);
           
    bool
    operator != (const lldb::SBBreakpoint& rhs);

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
    
    %pythoncode %{
        
        class locations_access(object):
            '''A helper object that will lazily hand out locations for a breakpoint when supplied an index.'''
            def __init__(self, sbbreakpoint):
                self.sbbreakpoint = sbbreakpoint
        
            def __len__(self):
                if self.sbbreakpoint:
                    return int(self.sbbreakpoint.GetNumLocations())
                return 0
        
            def __getitem__(self, key):
                if type(key) is int and key < len(self):
                    return self.sbbreakpoint.GetLocationAtIndex(key)
                return None
        
        def get_locations_access_object(self):
            '''An accessor function that returns a locations_access() object which allows lazy location access from a lldb.SBBreakpoint object.'''
            return self.locations_access (self)
        
        def get_breakpoint_location_list(self):
            '''An accessor function that returns a list() that contains all locations in a lldb.SBBreakpoint object.'''
            locations = []
            accessor = self.get_locations_access_object()
            for idx in range(len(accessor)):
                locations.append(accessor[idx])
            return locations
        
        __swig_getmethods__["locations"] = get_breakpoint_location_list
        if _newclass: locations = property(get_breakpoint_location_list, None, doc='''A read only property that returns a list() of lldb.SBBreakpointLocation objects for this breakpoint.''')
        
        __swig_getmethods__["location"] = get_locations_access_object
        if _newclass: location = property(get_locations_access_object, None, doc='''A read only property that returns an object that can access locations by index (not location ID) (location = bkpt.location[12]).''')

        __swig_getmethods__["id"] = GetID
        if _newclass: id = property(GetID, None, doc='''A read only property that returns the ID of this breakpoint.''')
            
        __swig_getmethods__["enabled"] = IsEnabled
        __swig_setmethods__["enabled"] = SetEnabled
        if _newclass: enabled = property(IsEnabled, SetEnabled, doc='''A read/write property that configures whether this breakpoint is enabled or not.''')

        __swig_getmethods__["one_shot"] = IsOneShot
        __swig_setmethods__["one_shot"] = SetOneShot
        if _newclass: one_shot = property(IsOneShot, SetOneShot, doc='''A read/write property that configures whether this breakpoint is one-shot (deleted when hit) or not.''')
            
        __swig_getmethods__["num_locations"] = GetNumLocations
        if _newclass: num_locations = property(GetNumLocations, None, doc='''A read only property that returns the count of locations of this breakpoint.''')

    %}

    
};

class SBBreakpointListImpl;

class LLDB_API SBBreakpointList
{
public:
  SBBreakpointList(SBTarget &target);
    
  ~SBBreakpointList();

  size_t GetSize() const;
  
  SBBreakpoint
  GetBreakpointAtIndex(size_t idx);
  
  SBBreakpoint
  FindBreakpointByID(lldb::break_id_t);

  void Append(const SBBreakpoint &sb_bkpt);

  bool AppendIfUnique(const SBBreakpoint &sb_bkpt);

  void AppendByID (lldb::break_id_t id);

  void Clear();
private:
  std::shared_ptr<SBBreakpointListImpl> m_opaque_sp;
};

} // namespace lldb
