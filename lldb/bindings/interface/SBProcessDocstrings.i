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
) lldb::SBProcess;

%feature("autodoc", "
    Writes data into the current process's stdin. API client specifies a Python
    string as the only argument."
) lldb::SBProcess::PutSTDIN;

%feature("autodoc", "
    Reads data from the current process's stdout stream. API client specifies
    the size of the buffer to read data into. It returns the byte buffer in a
    Python string."
) lldb::SBProcess::GetSTDOUT;

%feature("autodoc", "
    Reads data from the current process's stderr stream. API client specifies
    the size of the buffer to read data into. It returns the byte buffer in a
    Python string."
) lldb::SBProcess::GetSTDERR;

%feature("docstring", "
    Remote connection related functions. These will fail if the
    process is not in eStateConnected. They are intended for use
    when connecting to an externally managed debugserver instance."
) lldb::SBProcess::RemoteAttachToProcessWithID;

%feature("docstring",
"See SBTarget.Launch for argument description and usage."
) lldb::SBProcess::RemoteLaunch;

%feature("autodoc", "
    Returns the INDEX'th thread from the list of current threads.  The index
    of a thread is only valid for the current stop.  For a persistent thread
    identifier use either the thread ID or the IndexID.  See help on SBThread
    for more details."
) lldb::SBProcess::GetThreadAtIndex;

%feature("autodoc", "
    Returns the thread with the given thread ID."
) lldb::SBProcess::GetThreadByID;

%feature("autodoc", "
    Returns the thread with the given thread IndexID."
) lldb::SBProcess::GetThreadByIndexID;

%feature("autodoc", "
    Returns the currently selected thread."
) lldb::SBProcess::GetSelectedThread;

%feature("autodoc", "
    Lazily create a thread on demand through the current OperatingSystem plug-in, if the current OperatingSystem plug-in supports it."
) lldb::SBProcess::CreateOSPluginThread;

%feature("autodoc", "
    Returns the process ID of the process."
) lldb::SBProcess::GetProcessID;

%feature("autodoc", "
    Returns an integer ID that is guaranteed to be unique across all process instances. This is not the process ID, just a unique integer for comparison and caching purposes."
) lldb::SBProcess::GetUniqueID;

%feature("docstring", "
    Kills the process and shuts down all threads that were spawned to
    track and monitor process."
) lldb::SBProcess::Destroy;

%feature("docstring", "Same as Destroy(self).") lldb::SBProcess::Kill;

%feature("docstring", "Sends the process a unix signal.") lldb::SBProcess::Signal;

%feature("docstring", "
    Returns a stop id that will increase every time the process executes.  If
    include_expression_stops is true, then stops caused by expression evaluation
    will cause the returned value to increase, otherwise the counter returned will
    only increase when execution is continued explicitly by the user.  Note, the value
    will always increase, but may increase by more than one per stop."
) lldb::SBProcess::GetStopID;

%feature("autodoc", "
    Reads memory from the current process's address space and removes any
    traps that may have been inserted into the memory. It returns the byte
    buffer in a Python string. Example: ::

        # Read 4 bytes from address 'addr' and assume error.Success() is True.
        content = process.ReadMemory(addr, 4, error)
        new_bytes = bytearray(content)"
) lldb::SBProcess::ReadMemory;

%feature("autodoc", "
    Writes memory to the current process's address space and maintains any
    traps that might be present due to software breakpoints. Example: ::

        # Create a Python string from the byte array.
        new_value = str(bytes)
        result = process.WriteMemory(addr, new_value, error)
        if not error.Success() or result != len(bytes):
            print('SBProcess.WriteMemory() failed!')"
) lldb::SBProcess::WriteMemory;

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
            print('error: ', error)"
) lldb::SBProcess::ReadCStringFromMemory;


%feature("autodoc", "
    Reads an unsigned integer from memory given a byte size and an address.
    Returns the unsigned integer that was read. Example: ::

        # Read a 4 byte unsigned integer from address 0x1000
        error = lldb.SBError()
        uint = ReadUnsignedFromMemory(0x1000, 4, error)
        if error.Success():
            print('integer: %u' % uint)
        else
            print('error: ', error)"
) lldb::SBProcess::ReadUnsignedFromMemory;


%feature("autodoc", "
    Reads a pointer from memory from an address and returns the value. Example: ::

        # Read a pointer from address 0x1000
        error = lldb.SBError()
        ptr = ReadPointerFromMemory(0x1000, error)
        if error.Success():
            print('pointer: 0x%x' % ptr)
        else
            print('error: ', error)"
) lldb::SBProcess::ReadPointerFromMemory;


%feature("autodoc", "
    Returns the implementation object of the process plugin if available. None
    otherwise."
) lldb::SBProcess::GetScriptedImplementation;

%feature("autodoc", "
    Returns the process' extended crash information."
) lldb::SBProcess::GetExtendedCrashInformation;

%feature("autodoc", "
    Load the library whose filename is given by image_spec looking in all the
    paths supplied in the paths argument.  If successful, return a token that
    can be passed to UnloadImage and fill loaded_path with the path that was
    successfully loaded.  On failure, return
    lldb.LLDB_INVALID_IMAGE_TOKEN."
) lldb::SBProcess::LoadImageUsingPaths;

%feature("autodoc", "
    Return the number of different thread-origin extended backtraces
    this process can support as a uint32_t.
    When the process is stopped and you have an SBThread, lldb may be
    able to show a backtrace of when that thread was originally created,
    or the work item was enqueued to it (in the case of a libdispatch
    queue)."
) lldb::SBProcess::GetNumExtendedBacktraceTypes;

%feature("autodoc", "
    Takes an index argument, returns the name of one of the thread-origin
    extended backtrace methods as a str."
) lldb::SBProcess::GetExtendedBacktraceTypeAtIndex;

%feature("autodoc", "
    Get information about the process.
    Valid process info will only be returned when the process is alive,
    use IsValid() to check if the info returned is valid. ::

        process_info = process.GetProcessInfo()
        if process_info.IsValid():
            process_info.GetProcessID()"
) lldb::SBProcess::GetProcessInfo;

%feature("autodoc", "
    Allocates a block of memory within the process, with size and
    access permissions specified in the arguments. The permissions
    argument is an or-combination of zero or more of
    lldb.ePermissionsWritable, lldb.ePermissionsReadable, and
    lldb.ePermissionsExecutable. Returns the address
    of the allocated buffer in the process, or
    lldb.LLDB_INVALID_ADDRESS if the allocation failed."
) lldb::SBProcess::AllocateMemory;

%feature("autodoc", "
    Deallocates the block of memory (previously allocated using
    AllocateMemory) given in the argument."
) lldb::SBProcess::DeallocateMemory;
