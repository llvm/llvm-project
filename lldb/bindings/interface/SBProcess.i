//===-- SWIG Interface for SBProcess ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Represents the process associated with the target program.

SBProcess supports thread iteration. For example (from test/lldbutil.py), ::

    # ==================================================
    # Utility functions related to Threads and Processes
    # ==================================================

    def get_stopped_threads(process, reason):
        '''Returns the thread(s) with the specified stop reason in a list.

        The list can be empty if no such thread exists.
        '''
        threads = []
        for t in process:
            if t.GetStopReason() == reason:
                threads.append(t)
        return threads
"
) SBProcess;
class SBProcess
{
public:
    enum
    {
        eBroadcastBitStateChanged   = (1 << 0),
        eBroadcastBitInterrupt      = (1 << 1),
        eBroadcastBitSTDOUT         = (1 << 2),
        eBroadcastBitSTDERR         = (1 << 3),
        eBroadcastBitProfileData    = (1 << 4),
        eBroadcastBitStructuredData = (1 << 5)
    };

    SBProcess ();

    SBProcess (const lldb::SBProcess& rhs);

    ~SBProcess();

    static const char *
    GetBroadcasterClassName ();

    const char *
    GetPluginName ();

    const char *
    GetShortPluginName ();

    void
    Clear ();

    bool
    IsValid() const;

    explicit operator bool() const;

    lldb::SBTarget
    GetTarget() const;

    lldb::ByteOrder
    GetByteOrder() const;

    %feature("autodoc", "
    Writes data into the current process's stdin. API client specifies a Python
    string as the only argument.") PutSTDIN;
    size_t
    PutSTDIN (const char *src, size_t src_len);

    %feature("autodoc", "
    Reads data from the current process's stdout stream. API client specifies
    the size of the buffer to read data into. It returns the byte buffer in a
    Python string.") GetSTDOUT;
    size_t
    GetSTDOUT (char *dst, size_t dst_len) const;

    %feature("autodoc", "
    Reads data from the current process's stderr stream. API client specifies
    the size of the buffer to read data into. It returns the byte buffer in a
    Python string.") GetSTDERR;
    size_t
    GetSTDERR (char *dst, size_t dst_len) const;

    size_t
    GetAsyncProfileData(char *dst, size_t dst_len) const;

    void
    ReportEventState (const lldb::SBEvent &event, SBFile out) const;

    void
    ReportEventState (const lldb::SBEvent &event, FileSP BORROWED) const;

    void
    AppendEventStateReport (const lldb::SBEvent &event, lldb::SBCommandReturnObject &result);

    %feature("docstring", "
    Remote connection related functions. These will fail if the
    process is not in eStateConnected. They are intended for use
    when connecting to an externally managed debugserver instance.") RemoteAttachToProcessWithID;
    bool
    RemoteAttachToProcessWithID (lldb::pid_t pid,
                                 lldb::SBError& error);

    %feature("docstring",
    "See SBTarget.Launch for argument description and usage."
    ) RemoteLaunch;
    bool
    RemoteLaunch (char const **argv,
                  char const **envp,
                  const char *stdin_path,
                  const char *stdout_path,
                  const char *stderr_path,
                  const char *working_directory,
                  uint32_t launch_flags,
                  bool stop_at_entry,
                  lldb::SBError& error);

    //------------------------------------------------------------------
    // Thread related functions
    //------------------------------------------------------------------
    uint32_t
    GetNumThreads ();

    %feature("autodoc", "
    Returns the INDEX'th thread from the list of current threads.  The index
    of a thread is only valid for the current stop.  For a persistent thread
    identifier use either the thread ID or the IndexID.  See help on SBThread
    for more details.") GetThreadAtIndex;
    lldb::SBThread
    GetThreadAtIndex (size_t index);

    %feature("autodoc", "
    Returns the thread with the given thread ID.") GetThreadByID;
    lldb::SBThread
    GetThreadByID (lldb::tid_t sb_thread_id);

    %feature("autodoc", "
    Returns the thread with the given thread IndexID.") GetThreadByIndexID;
    lldb::SBThread
    GetThreadByIndexID (uint32_t index_id);

    %feature("autodoc", "
    Returns the currently selected thread.") GetSelectedThread;
    lldb::SBThread
    GetSelectedThread () const;

    %feature("autodoc", "
    Lazily create a thread on demand through the current OperatingSystem plug-in, if the current OperatingSystem plug-in supports it.") CreateOSPluginThread;
    lldb::SBThread
    CreateOSPluginThread (lldb::tid_t tid, lldb::addr_t context);

    bool
    SetSelectedThread (const lldb::SBThread &thread);

    bool
    SetSelectedThreadByID (lldb::tid_t tid);

    bool
    SetSelectedThreadByIndexID (uint32_t index_id);

    //------------------------------------------------------------------
    // Queue related functions
    //------------------------------------------------------------------
    uint32_t
    GetNumQueues ();

    lldb::SBQueue
    GetQueueAtIndex (uint32_t index);

    //------------------------------------------------------------------
    // Stepping related functions
    //------------------------------------------------------------------

    lldb::StateType
    GetState ();

    int
    GetExitStatus ();

    const char *
    GetExitDescription ();

    %feature("autodoc", "
    Returns the process ID of the process.") GetProcessID;
    lldb::pid_t
    GetProcessID ();

    %feature("autodoc", "
    Returns an integer ID that is guaranteed to be unique across all process instances. This is not the process ID, just a unique integer for comparison and caching purposes.") GetUniqueID;
    uint32_t
    GetUniqueID();

    uint32_t
    GetAddressByteSize() const;

    %feature("docstring", "
    Kills the process and shuts down all threads that were spawned to
    track and monitor process.") Destroy;
    lldb::SBError
    Destroy ();

    lldb::SBError
    Continue ();

    lldb::SBError
    Stop ();

    %feature("docstring", "Same as Destroy(self).") Destroy;
    lldb::SBError
    Kill ();

    lldb::SBError
    Detach ();

    %feature("docstring", "Sends the process a unix signal.") Signal;
    lldb::SBError
    Signal (int signal);

    lldb::SBUnixSignals
    GetUnixSignals();

    %feature("docstring", "
    Returns a stop id that will increase every time the process executes.  If
    include_expression_stops is true, then stops caused by expression evaluation
    will cause the returned value to increase, otherwise the counter returned will
    only increase when execution is continued explicitly by the user.  Note, the value
    will always increase, but may increase by more than one per stop.") GetStopID;
    uint32_t
    GetStopID(bool include_expression_stops = false);

    void
    SendAsyncInterrupt();

    %feature("autodoc", "
    Reads memory from the current process's address space and removes any
    traps that may have been inserted into the memory. It returns the byte
    buffer in a Python string. Example: ::

        # Read 4 bytes from address 'addr' and assume error.Success() is True.
        content = process.ReadMemory(addr, 4, error)
        new_bytes = bytearray(content)") ReadMemory;
    size_t
    ReadMemory (addr_t addr, void *buf, size_t size, lldb::SBError &error);

    %feature("autodoc", "
    Writes memory to the current process's address space and maintains any
    traps that might be present due to software breakpoints. Example: ::

        # Create a Python string from the byte array.
        new_value = str(bytes)
        result = process.WriteMemory(addr, new_value, error)
        if not error.Success() or result != len(bytes):
            print('SBProcess.WriteMemory() failed!')") WriteMemory;
    size_t
    WriteMemory (addr_t addr, const void *buf, size_t size, lldb::SBError &error);

    %feature("autodoc", "
    Reads a NULL terminated C string from the current process's address space.
    It returns a python string of the exact length, or truncates the string if
    the maximum character limit is reached. Example: ::

        # Read a C string of at most 256 bytes from address '0x1000'
        error = lldb.SBError()
        cstring = process.ReadCStringFromMemory(0x1000, 256, error)
        if error.Success():
            print('cstring: ', cstring)
        else
            print('error: ', error)") ReadCStringFromMemory;

    size_t
    ReadCStringFromMemory (addr_t addr, void *char_buf, size_t size, lldb::SBError &error);

    %feature("autodoc", "
    Reads an unsigned integer from memory given a byte size and an address.
    Returns the unsigned integer that was read. Example: ::

        # Read a 4 byte unsigned integer from address 0x1000
        error = lldb.SBError()
        uint = ReadUnsignedFromMemory(0x1000, 4, error)
        if error.Success():
            print('integer: %u' % uint)
        else
            print('error: ', error)") ReadUnsignedFromMemory;

    uint64_t
    ReadUnsignedFromMemory (addr_t addr, uint32_t byte_size, lldb::SBError &error);

    %feature("autodoc", "
    Reads a pointer from memory from an address and returns the value. Example: ::

        # Read a pointer from address 0x1000
        error = lldb.SBError()
        ptr = ReadPointerFromMemory(0x1000, error)
        if error.Success():
            print('pointer: 0x%x' % ptr)
        else
            print('error: ', error)") ReadPointerFromMemory;

    lldb::addr_t
    ReadPointerFromMemory (addr_t addr, lldb::SBError &error);


    // Events
    static lldb::StateType
    GetStateFromEvent (const lldb::SBEvent &event);

    static bool
    GetRestartedFromEvent (const lldb::SBEvent &event);

    static size_t
    GetNumRestartedReasonsFromEvent (const lldb::SBEvent &event);

    static const char *
    GetRestartedReasonAtIndexFromEvent (const lldb::SBEvent &event, size_t idx);

    static lldb::SBProcess
    GetProcessFromEvent (const lldb::SBEvent &event);

    static bool
    GetInterruptedFromEvent (const lldb::SBEvent &event);

    static lldb::SBStructuredData
    GetStructuredDataFromEvent (const lldb::SBEvent &event);

    static bool
    EventIsProcessEvent (const lldb::SBEvent &event);

    static bool
    EventIsStructuredDataEvent (const lldb::SBEvent &event);

    lldb::SBBroadcaster
    GetBroadcaster () const;

    bool
    GetDescription (lldb::SBStream &description);

    %feature("autodoc", "
    Returns the implementation object of the process plugin if available. None
    otherwise.") GetScriptedImplementation;
    ScriptedObject
    GetScriptedImplementation();

    %feature("autodoc", "
    Returns the process' extended crash information.") GetExtendedCrashInformation;
    lldb::SBStructuredData
    GetExtendedCrashInformation ();

    uint32_t
    GetNumSupportedHardwareWatchpoints (lldb::SBError &error) const;

    uint32_t
    LoadImage (lldb::SBFileSpec &image_spec, lldb::SBError &error);

    %feature("autodoc", "
    Load the library whose filename is given by image_spec looking in all the
    paths supplied in the paths argument.  If successful, return a token that
    can be passed to UnloadImage and fill loaded_path with the path that was
    successfully loaded.  On failure, return
    lldb.LLDB_INVALID_IMAGE_TOKEN.") LoadImageUsingPaths;
    uint32_t
    LoadImageUsingPaths(const lldb::SBFileSpec &image_spec,
                        SBStringList &paths,
                        lldb::SBFileSpec &loaded_path,
                        SBError &error);

    lldb::SBError
    UnloadImage (uint32_t image_token);

    lldb::SBError
    SendEventData (const char *event_data);

    %feature("autodoc", "
    Return the number of different thread-origin extended backtraces
    this process can support as a uint32_t.
    When the process is stopped and you have an SBThread, lldb may be
    able to show a backtrace of when that thread was originally created,
    or the work item was enqueued to it (in the case of a libdispatch
    queue).") GetNumExtendedBacktraceTypes;

    uint32_t
    GetNumExtendedBacktraceTypes ();

    %feature("autodoc", "
    Takes an index argument, returns the name of one of the thread-origin
    extended backtrace methods as a str.") GetExtendedBacktraceTypeAtIndex;

    const char *
    GetExtendedBacktraceTypeAtIndex (uint32_t idx);

    lldb::SBThreadCollection
    GetHistoryThreads (addr_t addr);

    bool
    IsInstrumentationRuntimePresent(lldb::InstrumentationRuntimeType type);

    lldb::SBError
    SaveCore(const char *file_name, const char *flavor, lldb::SaveCoreStyle core_style);

    lldb::SBError
    SaveCore(const char *file_name);

    lldb::SBError
    GetMemoryRegionInfo(lldb::addr_t load_addr, lldb::SBMemoryRegionInfo &region_info);

    lldb::SBMemoryRegionInfoList
    GetMemoryRegions();

    %feature("autodoc", "
    Get information about the process.
    Valid process info will only be returned when the process is alive,
    use IsValid() to check if the info returned is valid. ::

        process_info = process.GetProcessInfo()
        if process_info.IsValid():
            process_info.GetProcessID()") GetProcessInfo;
    lldb::SBProcessInfo
    GetProcessInfo();

    %feature("autodoc", "
    Allocates a block of memory within the process, with size and
    access permissions specified in the arguments. The permissions
    argument is an or-combination of zero or more of
    lldb.ePermissionsWritable, lldb.ePermissionsReadable, and
    lldb.ePermissionsExecutable. Returns the address
    of the allocated buffer in the process, or
    lldb.LLDB_INVALID_ADDRESS if the allocation failed.") AllocateMemory;

    lldb::addr_t
    AllocateMemory(size_t size, uint32_t permissions, lldb::SBError &error);

    %feature("autodoc", "
    Deallocates the block of memory (previously allocated using
    AllocateMemory) given in the argument.") DeallocateMemory;

    lldb::SBError
    DeallocateMemory(lldb::addr_t ptr);

    STRING_EXTENSION(SBProcess)

#ifdef SWIGPYTHON
    %pythoncode %{
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
                if type(key) is int and key < len(self):
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

};

}  // namespace lldb
