%extend lldb::SBAddressRange {
#ifdef SWIGPYTHON
    %pythoncode%{
      def __repr__(self):
        import lldb
        stream = lldb.SBStream()
        self.GetDescription(stream, lldb.target if lldb.target else lldb.SBTarget())
        return stream.GetData()
    %}
#endif
}
