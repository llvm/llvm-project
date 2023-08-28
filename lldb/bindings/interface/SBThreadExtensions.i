STRING_EXTENSION_OUTSIDE(SBThread)

%extend lldb::SBThread {
#ifdef SWIGPYTHON
    %pythoncode %{
        def __eq__(self, other):
            return not self.__ne__(other)

        def __int__(self):
            pass

        def __hex__(self):
            pass

        def __oct__(self):
            pass

        def __iter__(self):
            '''Iterate over all frames in a lldb.SBThread object.'''
            return lldb_iter(self, 'GetNumFrames', 'GetFrameAtIndex')

        def __len__(self):
            '''Return the number of frames in a lldb.SBThread object.'''
            return self.GetNumFrames()

        class frames_access(object):
            '''A helper object that will lazily hand out frames for a thread when supplied an index.'''
            def __init__(self, sbthread):
                self.sbthread = sbthread

            def __len__(self):
                if self.sbthread:
                    return int(self.sbthread.GetNumFrames())
                return 0

            def __getitem__(self, key):
                if isinstance(key, int):
                    count = len(self)
                    if -count <= key < count:
                        key %= count
                        return self.sbthread.GetFrameAtIndex(key)
                return None

        def get_frames_access_object(self):
            '''An accessor function that returns a frames_access() object which allows lazy frame access from a lldb.SBThread object.'''
            return self.frames_access (self)

        def get_thread_frames(self):
            '''An accessor function that returns a list() that contains all frames in a lldb.SBThread object.'''
            frames = []
            for frame in self:
                frames.append(frame)
            return frames

        id = property(GetThreadID, None, doc='''A read only property that returns the thread ID as an integer.''')
        idx = property(GetIndexID, None, doc='''A read only property that returns the thread index ID as an integer. Thread index ID values start at 1 and increment as threads come and go and can be used to uniquely identify threads.''')
        return_value = property(GetStopReturnValue, None, doc='''A read only property that returns an lldb object that represents the return value from the last stop (lldb.SBValue) if we just stopped due to stepping out of a function.''')
        process = property(GetProcess, None, doc='''A read only property that returns an lldb object that represents the process (lldb.SBProcess) that owns this thread.''')
        num_frames = property(GetNumFrames, None, doc='''A read only property that returns the number of stack frames in this thread as an integer.''')
        frames = property(get_thread_frames, None, doc='''A read only property that returns a list() of lldb.SBFrame objects for all frames in this thread.''')
        frame = property(get_frames_access_object, None, doc='''A read only property that returns an object that can be used to access frames as an array ("frame_12 = lldb.thread.frame[12]").''')
        name = property(GetName, None, doc='''A read only property that returns the name of this thread as a string.''')
        queue = property(GetQueueName, None, doc='''A read only property that returns the dispatch queue name of this thread as a string.''')
        queue_id = property(GetQueueID, None, doc='''A read only property that returns the dispatch queue id of this thread as an integer.''')
        stop_reason = property(GetStopReason, None, doc='''A read only property that returns an lldb enumeration value (see enumerations that start with "lldb.eStopReason") that represents the reason this thread stopped.''')
        is_suspended = property(IsSuspended, None, doc='''A read only property that returns a boolean value that indicates if this thread is suspended.''')
        is_stopped = property(IsStopped, None, doc='''A read only property that returns a boolean value that indicates if this thread is stopped but not exited.''')
    %}
#endif
}
