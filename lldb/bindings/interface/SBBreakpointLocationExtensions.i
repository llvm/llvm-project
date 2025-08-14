STRING_EXTENSION_LEVEL_OUTSIDE(SBBreakpointLocation, lldb::eDescriptionLevelFull)

%extend lldb::SBBreakpointLocation {
#ifdef SWIGPYTHON
    %pythoncode%{
    # operator== is a free function, which swig does not handle, so we inject
    # our own equality operator here
    def __eq__(self, other):
      return not self.__ne__(other)

    addr = property(GetAddress, doc='A read only property that returns the address of this breakpoint location.')
    auto_continue = property(GetAutoContinue, SetAutoContinue, doc='A read/write property that configures the auto-continue property of this breakpoint location.')
    breakpoint = property(GetBreakpoint, doc='A read only property that returns the parent breakpoint of this breakpoint location.')
    condition = property(GetCondition, SetCondition, doc='A read/write property that configures the condition of this breakpoint location.')
    hit_count = property(GetHitCount, doc='A read only property that returns the hit count of this breakpoint location.')
    id = property(GetID, doc='A read only property that returns the id of this breakpoint location.')
    ignore_count = property(GetIgnoreCount, SetIgnoreCount, doc='A read/write property that configures the ignore count of this breakpoint location.')
    load_addr = property(GetLoadAddress, doc='A read only property that returns the load address of this breakpoint location.')
    queue_name = property(GetQueueName, SetQueueName, doc='A read/write property that configures the queue name criteria of this breakpoint location.')
    thread_id = property(GetThreadID, SetThreadID, doc='A read/write property that configures the thread id criteria of this breakpoint location.')
    thread_index = property(GetThreadIndex, SetThreadIndex, doc='A read/write property that configures the thread index criteria of this breakpoint location.')
    thread_name = property(GetThreadName, SetThreadName, doc='A read/write property that configures the thread name criteria of this breakpoint location.')
    %}
#endif
}
