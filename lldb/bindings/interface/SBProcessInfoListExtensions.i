%extend lldb::SBProcessInfoList {
#ifdef SWIGPYTHON
    %pythoncode%{
    def __len__(self):
      '''Return the number of process info in a lldb.SBProcessInfoListExtensions object.'''
      return self.GetSize()

    def __iter__(self):
      '''Iterate over all the process info in a lldb.SBProcessInfoListExtensions object.'''
      return lldb_iter(self, 'GetSize', 'GetProcessInfoAtIndex')
    %}
#endif
}
