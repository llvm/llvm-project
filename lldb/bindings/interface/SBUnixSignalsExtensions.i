%extend lldb::SBUnixSignals {
#ifdef SWIGPYTHON
    %pythoncode %{
        def __iter__(self):
            '''Iterate over all signals in a lldb.SBUnixSignals object.'''
            for i in range(self.GetNumSignals()):
                yield self.GetSignalAtIndex(i)

        def __len__(self):
            return self.GetNumSignals()

        def get_unix_signals_list(self):
            signals = []
            for idx in range(0, self.GetNumSignals()):
                signals.append(self.GetSignalAtIndex(idx))
            return signals

        threads = property(get_unix_signals_list, None, doc='''A read only property that returns a list() of valid signal numbers for this platform.''')
    %}
#endif
}
