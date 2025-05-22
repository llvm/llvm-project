%extend lldb::SBUnixSignals {
#ifdef SWIGPYTHON
    %pythoncode %{
        def __iter__(self):
            '''Iterate over all signals in a lldb.SBUnixSignals object.'''
            return lldb_iter(self, 'GetNumSignals', 'GetSignalAtIndex')

        def __len__(self):
            return self.GetNumSignals()

        def get_unix_signals_list(self):
            signals = []
            for idx in range(0, self.GetNumSignals()):
                signals.append(self.GetSignalAtIndex(sig))
            return signals

        threads = property(get_unix_signals_list, None, doc='''A read only property that returns a list() of valid signal numbers for this platform.''')
    %}
#endif
}
