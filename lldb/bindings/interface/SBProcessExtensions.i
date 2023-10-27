STRING_EXTENSION_OUTSIDE(SBProcess)
%extend lldb::SBProcess {
#ifdef SWIGPYTHON
    %pythoncode %{
        def WriteMemoryAsCString(self, addr, str, error):
            '''
              WriteMemoryAsCString(self, addr, str, error):
                This functions the same as `WriteMemory` except a null-terminator is appended
                to the end of the buffer if it is not there already.
            '''
            if not str or len(str) == 0:
                return 0
            if not str[-1] == '\0':
                str += '\0'
            return self.WriteMemory(addr, str, error)

        def __get_is_alive__(self):
            '''Returns "True" if the process is currently alive, "False" otherwise'''
            s = self.GetState()
            if (s == eStateAttaching or
                s == eStateLaunching or
                s == eStateStopped or
                s == eStateRunning or
                s == eStateStepping or
                s == eStateCrashed or
                s == eStateSuspended):
                return True
            return False

        def __get_is_running__(self):
            '''Returns "True" if the process is currently running, "False" otherwise'''
            state = self.GetState()
            if state == eStateRunning or state == eStateStepping:
                return True
            return False

        def __get_is_stopped__(self):
            '''Returns "True" if the process is currently stopped, "False" otherwise'''
            state = self.GetState()
            if state == eStateStopped or state == eStateCrashed or state == eStateSuspended:
                return True
            return False

        class threads_access(object):
            '''A helper object that will lazily hand out thread for a process when supplied an index.'''
            def __init__(self, sbprocess):
                self.sbprocess = sbprocess

            def __len__(self):
                if self.sbprocess:
                    return int(self.sbprocess.GetNumThreads())
                return 0

            def __getitem__(self, key):
                if isinstance(key, int):
                    count = len(self)
                    if -count <= key < count:
                        key %= count
                        return self.sbprocess.GetThreadAtIndex(key)
                return None

        def get_threads_access_object(self):
            '''An accessor function that returns a modules_access() object which allows lazy thread access from a lldb.SBProcess object.'''
            return self.threads_access (self)

        def get_process_thread_list(self):
            '''An accessor function that returns a list() that contains all threads in a lldb.SBProcess object.'''
            threads = []
            accessor = self.get_threads_access_object()
            for idx in range(len(accessor)):
                threads.append(accessor[idx])
            return threads

        def __iter__(self):
            '''Iterate over all threads in a lldb.SBProcess object.'''
            return lldb_iter(self, 'GetNumThreads', 'GetThreadAtIndex')

        def __len__(self):
            '''Return the number of threads in a lldb.SBProcess object.'''
            return self.GetNumThreads()

        def __int__(self):
            return self.GetProcessID()

        threads = property(get_process_thread_list, None, doc='''A read only property that returns a list() of lldb.SBThread objects for this process.''')
        thread = property(get_threads_access_object, None, doc='''A read only property that returns an object that can access threads by thread index (thread = lldb.process.thread[12]).''')
        is_alive = property(__get_is_alive__, None, doc='''A read only property that returns a boolean value that indicates if this process is currently alive.''')
        is_running = property(__get_is_running__, None, doc='''A read only property that returns a boolean value that indicates if this process is currently running.''')
        is_stopped = property(__get_is_stopped__, None, doc='''A read only property that returns a boolean value that indicates if this process is currently stopped.''')
        id = property(GetProcessID, None, doc='''A read only property that returns the process ID as an integer.''')
        target = property(GetTarget, None, doc='''A read only property that an lldb object that represents the target (lldb.SBTarget) that owns this process.''')
        num_threads = property(GetNumThreads, None, doc='''A read only property that returns the number of threads in this process as an integer.''')
        selected_thread = property(GetSelectedThread, SetSelectedThread, doc='''A read/write property that gets/sets the currently selected thread in this process. The getter returns a lldb.SBThread object and the setter takes an lldb.SBThread object.''')
        state = property(GetState, None, doc='''A read only property that returns an lldb enumeration value (see enumerations that start with "lldb.eState") that represents the current state of this process (running, stopped, exited, etc.).''')
        exit_state = property(GetExitStatus, None, doc='''A read only property that returns an exit status as an integer of this process when the process state is lldb.eStateExited.''')
        exit_description = property(GetExitDescription, None, doc='''A read only property that returns an exit description as a string of this process when the process state is lldb.eStateExited.''')
        broadcaster = property(GetBroadcaster, None, doc='''A read only property that an lldb object that represents the broadcaster (lldb.SBBroadcaster) for this process.''')
    %}
#endif
}
