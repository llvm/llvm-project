%extend lldb::SBUnixSignals {
#ifdef SWIGPYTHON
    %pythoncode %{
        def get_unix_signals_list(self):
            signals = []
            for idx in range(0, self.GetNumSignals()):
                signals.append(self.GetSignalAtIndex(sig))
            return signals

        threads = property(get_unix_signals_list, None, doc='''A read only property that returns a list() of valid signal numbers for this platform.''')
    %}
#endif
}
