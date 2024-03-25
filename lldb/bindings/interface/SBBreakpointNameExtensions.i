STRING_EXTENSION_OUTSIDE(SBBreakpointName)

%extend lldb::SBBreakpointName {
#ifdef SWIGPYTHON
    %pythoncode%{
    # operator== is a free function, which swig does not handle, so we inject
    # our own equality operator here
    def __eq__(self, other):
      return not self.__ne__(other)
    %}
#endif
}
