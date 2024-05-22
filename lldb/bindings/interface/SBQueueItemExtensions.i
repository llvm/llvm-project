%extend lldb::SBQueueItem {
#ifdef SWIGPYTHON
    %pythoncode%{
    def __hex__(self):
      return self.GetAddress()
    %}
#endif
}
