STRING_EXTENSION_OUTSIDE(SBMemoryRegionInfo)

%extend lldb::SBMemoryRegionInfo {
#ifdef SWIGPYTHON
    %pythoncode%{
    # operator== is a free function, which swig does not handle, so we inject
    # our own equality operator here
    def __eq__(self, other):
      return not self.__ne__(other)

    def __hex__(self):
      return self.GetRegionBase()

    def __len__(self):
      return self.GetRegionEnd() - self.GetRegionBase()
    %}
#endif
}
