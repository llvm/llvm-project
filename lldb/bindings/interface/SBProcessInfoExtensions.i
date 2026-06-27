
%extend lldb::SBProcessInfo {

#ifdef SWIGPYTHON
    %pythoncode %{ 
    @property
    def arguments(self):
        """A read only property that returns a list of lldb.SBProcessInfo arguments."""
        if not hasattr(self, "_arguments"):
            self._arguments = [
                self.GetArgumentAtIndex(idx)
                for idx in range(self.GetNumArguments())
            ]
            
        return self._arguments
    %}
#endif
}
