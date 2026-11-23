STRING_EXTENSION_OUTSIDE(SBBreakpointName)

%extend lldb::SBBreakpointName {
#ifdef SWIGPYTHON
    %pythoncode%{
    # operator== is a free function, which swig does not handle, so we inject
    # our own equality operator here
    def __eq__(self, other):
      return not self.__ne__(other)

    auto_continue = property(GetAutoContinue, SetAutoContinue, doc='A read/write property that configures the auto-continue property of this breakpoint name.')
    condition = property(GetCondition, SetCondition, doc='A read/write property that configures the condition of this breakpoint name.')
    enabled = property(IsEnabled, SetEnabled, doc='''A read/write property that configures whether this breakpoint name is enabled or not.''')
    ignore_count = property(GetIgnoreCount, SetIgnoreCount, doc='A read/write property that configures the ignore count of this breakpoint name.')
    one_shot = property(IsOneShot, SetOneShot, doc='''A read/write property that configures whether this breakpoint name is one-shot (deleted when hit) or not.''')
    queue_name = property(GetQueueName, SetQueueName, doc='A read/write property that configures the queue name criteria of this breakpoint name.')
    thread_id = property(GetThreadID, SetThreadID, doc='A read/write property that configures the thread id criteria of this breakpoint name.')
    thread_index = property(GetThreadIndex, SetThreadIndex, doc='A read/write property that configures the thread index criteria of this breakpoint name.')
    thread_name = property(GetThreadName, SetThreadName, doc='A read/write property that configures the thread name criteria of this breakpoint name.')
    %}
#endif
}
