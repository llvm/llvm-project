%feature("docstring",
"Represents the target program running under the debugger.

SBTarget supports module, breakpoint, and watchpoint iterations. For example, ::

    for m in target.module_iter():
        print m

produces: ::

    (x86_64) /Volumes/data/lldb/svn/trunk/test/python_api/lldbutil/iter/a.out
    (x86_64) /usr/lib/dyld
    (x86_64) /usr/lib/libstdc++.6.dylib
    (x86_64) /usr/lib/libSystem.B.dylib
    (x86_64) /usr/lib/system/libmathCommon.A.dylib
    (x86_64) /usr/lib/libSystem.B.dylib(__commpage)

and, ::

    for b in target.breakpoint_iter():
        print b

produces: ::

    SBBreakpoint: id = 1, file ='main.cpp', line = 66, locations = 1
    SBBreakpoint: id = 2, file ='main.cpp', line = 85, locations = 1

and, ::

    for wp_loc in target.watchpoint_iter():
        print wp_loc

produces: ::

    Watchpoint 1: addr = 0x1034ca048 size = 4 state = enabled type = rw
        declare @ '/Volumes/data/lldb/svn/trunk/test/python_api/watchpoint/main.c:12'
        hw_index = 0  hit_count = 2     ignore_count = 0"
) lldb::SBTarget;

%feature("docstring", "
    Return the platform object associated with the target.

    After return, the platform object should be checked for
    validity.

    @return
        A platform object."
) lldb::SBTarget::GetPlatform;

%feature("docstring", "
    Install any binaries that need to be installed.

    This function does nothing when debugging on the host system.
    When connected to remote platforms, the target's main executable
    and any modules that have their install path set will be
    installed on the remote platform. If the main executable doesn't
    have an install location set, it will be installed in the remote
    platform's working directory.

    @return
        An error describing anything that went wrong during
        installation."
) lldb::SBTarget::Install;

%feature("docstring", "
    Launch a new process.

    Launch a new process by spawning a new process using the
    target object's executable module's file as the file to launch.
    Arguments are given in argv, and the environment variables
    are in envp. Standard input and output files can be
    optionally re-directed to stdin_path, stdout_path, and
    stderr_path.

    @param[in] listener
        An optional listener that will receive all process events.
        If listener is valid then listener will listen to all
        process events. If not valid, then this target's debugger
        (SBTarget::GetDebugger()) will listen to all process events.

    @param[in] argv
        The argument array.

    @param[in] envp
        The environment array.

    @param[in] launch_flags
        Flags to modify the launch (@see lldb::LaunchFlags)

    @param[in] stdin_path
        The path to use when re-directing the STDIN of the new
        process. If all stdXX_path arguments are NULL, a pseudo
        terminal will be used.

    @param[in] stdout_path
        The path to use when re-directing the STDOUT of the new
        process. If all stdXX_path arguments are NULL, a pseudo
        terminal will be used.

    @param[in] stderr_path
        The path to use when re-directing the STDERR of the new
        process. If all stdXX_path arguments are NULL, a pseudo
        terminal will be used.

    @param[in] working_directory
        The working directory to have the child process run in

    @param[in] launch_flags
        Some launch options specified by logical OR'ing
        lldb::LaunchFlags enumeration values together.

    @param[in] stop_at_entry
        If false do not stop the inferior at the entry point.

    @param[out]
        An error object. Contains the reason if there is some failure.

    @return
         A process object for the newly created process.

    For example,

        process = target.Launch(self.dbg.GetListener(), None, None,
                                None, '/tmp/stdout.txt', None,
                                None, 0, False, error)

    launches a new process by passing nothing for both the args and the envs
    and redirect the standard output of the inferior to the /tmp/stdout.txt
    file. It does not specify a working directory so that the debug server
    will use its idea of what the current working directory is for the
    inferior. Also, we ask the debugger not to stop the inferior at the
    entry point. If no breakpoint is specified for the inferior, it should
    run to completion if no user interaction is required."
) lldb::SBTarget::Launch;

%feature("docstring", "
    Launch a new process with sensible defaults.

    :param argv: The argument array.
    :param envp: The environment array.
    :param working_directory: The working directory to have the child process run in
    :return: The newly created process.
    :rtype: SBProcess

    A pseudo terminal will be used as stdin/stdout/stderr.
    No launch flags are passed and the target's debuger is used as a listener.

    For example, ::

        process = target.LaunchSimple(['X', 'Y', 'Z'], None, os.getcwd())

    launches a new process by passing 'X', 'Y', 'Z' as the args to the
    executable."
) lldb::SBTarget::LaunchSimple;

%feature("docstring", "
    Load a core file

    @param[in] core_file
        File path of the core dump.

    @param[out] error
        An error explaining what went wrong if the operation fails.
        (Optional)

    @return
         A process object for the newly created core file.

    For example,

        process = target.LoadCore('./a.out.core')

    loads a new core file and returns the process object."
) lldb::SBTarget::LoadCore;

%feature("docstring", "
    Attach to process with pid.

    @param[in] listener
        An optional listener that will receive all process events.
        If listener is valid then listener will listen to all
        process events. If not valid, then this target's debugger
        (SBTarget::GetDebugger()) will listen to all process events.

    @param[in] pid
        The process ID to attach to.

    @param[out]
        An error explaining what went wrong if attach fails.

    @return
         A process object for the attached process."
) lldb::SBTarget::AttachToProcessWithID;

%feature("docstring", "
    Attach to process with name.

    @param[in] listener
        An optional listener that will receive all process events.
        If listener is valid then listener will listen to all
        process events. If not valid, then this target's debugger
        (SBTarget::GetDebugger()) will listen to all process events.

    @param[in] name
        Basename of process to attach to.

    @param[in] wait_for
        If true wait for a new instance of 'name' to be launched.

    @param[out]
        An error explaining what went wrong if attach fails.

    @return
         A process object for the attached process."
) lldb::SBTarget::AttachToProcessWithName;

%feature("docstring", "
    Connect to a remote debug server with url.

    @param[in] listener
        An optional listener that will receive all process events.
        If listener is valid then listener will listen to all
        process events. If not valid, then this target's debugger
        (SBTarget::GetDebugger()) will listen to all process events.

    @param[in] url
        The url to connect to, e.g., 'connect://localhost:12345'.

    @param[in] plugin_name
        The plugin name to be used; can be NULL.

    @param[out]
        An error explaining what went wrong if the connect fails.

    @return
         A process object for the connected process."
) lldb::SBTarget::ConnectRemote;

%feature("docstring", "
    Append the path mapping (from -> to) to the target's paths mapping list."
) lldb::SBTarget::AppendImageSearchPath;

%feature("docstring", "
    Find compile units related to this target and passed source
    file.

    :param sb_file_spec: A :py:class:`lldb::SBFileSpec` object that contains source file
        specification.
    :return: The symbol contexts for all the matches.
    :rtype: SBSymbolContextList"
) lldb::SBTarget::FindCompileUnits;

%feature("docstring", "
    Architecture data byte width accessor

    :return: The size in 8-bit (host) bytes of a minimum addressable unit from the Architecture's data bus.

    "
) lldb::SBTarget::GetDataByteSize;

%feature("docstring", "
    Architecture code byte width accessor.

    :return: The size in 8-bit (host) bytes of a minimum addressable unit from the Architecture's code bus.

    "
) lldb::SBTarget::GetCodeByteSize;

%feature("docstring", "
    Find functions by name.

    :param name: The name of the function we are looking for.

    :param name_type_mask:
        A logical OR of one or more FunctionNameType enum bits that
        indicate what kind of names should be used when doing the
        lookup. Bits include fully qualified names, base names,
        C++ methods, or ObjC selectors.
        See FunctionNameType for more details.

    :return:
        A lldb::SBSymbolContextList that gets filled in with all of
        the symbol contexts for all the matches."
) lldb::SBTarget::FindFunctions;

%feature("docstring", "
    Find global and static variables by name.

    @param[in] name
        The name of the global or static variable we are looking
        for.

    @param[in] max_matches
        Allow the number of matches to be limited to max_matches.

    @return
        A list of matched variables in an SBValueList."
) lldb::SBTarget::FindGlobalVariables;

 %feature("docstring", "
    Find the first global (or static) variable by name.

    @param[in] name
        The name of the global or static variable we are looking
        for.

    @return
        An SBValue that gets filled in with the found variable (if any)."
) lldb::SBTarget::FindFirstGlobalVariable;

%feature("docstring", "
    Resolve a current file address into a section offset address.

    @param[in] file_addr

    @return
        An SBAddress which will be valid if..."
) lldb::SBTarget::ResolveFileAddress;

%feature("docstring", "
    Read target memory. If a target process is running then memory
    is read from here. Otherwise the memory is read from the object
    files. For a target whose bytes are sized as a multiple of host
    bytes, the data read back will preserve the target's byte order.

    @param[in] addr
        A target address to read from.

    @param[out] buf
        The buffer to read memory into.

    @param[in] size
        The maximum number of host bytes to read in the buffer passed
        into this call

    @param[out] error
        Error information is written here if the memory read fails.

    @return
        The amount of data read in host bytes."
) lldb::SBTarget::ReadMemory;

%feature("docstring", "
    Create a breakpoint using a scripted resolver.

    @param[in] class_name
       This is the name of the class that implements a scripted resolver.
       The class should have the following signature: ::

           class Resolver:
               def __init__(self, bkpt, extra_args):
                   # bkpt - the breakpoint for which this is the resolver.  When
                   # the resolver finds an interesting address, call AddLocation
                   # on this breakpoint to add it.
                   #
                   # extra_args - an SBStructuredData that can be used to
                   # parametrize this instance.  Same as the extra_args passed
                   # to BreakpointCreateFromScript.

               def __get_depth__ (self):
                   # This is optional, but if defined, you should return the
                   # depth at which you want the callback to be called.  The
                   # available options are:
                   #    lldb.eSearchDepthModule
                   #    lldb.eSearchDepthCompUnit
                   # The default if you don't implement this method is
                   # eSearchDepthModule.

               def __callback__(self, sym_ctx):
                   # sym_ctx - an SBSymbolContext that is the cursor in the
                   # search through the program to resolve breakpoints.
                   # The sym_ctx will be filled out to the depth requested in
                   # __get_depth__.
                   # Look in this sym_ctx for new breakpoint locations,
                   # and if found use bkpt.AddLocation to add them.
                   # Note, you will only get called for modules/compile_units that
                   # pass the SearchFilter provided by the module_list & file_list
                   # passed into BreakpointCreateFromScript.

               def get_short_help(self):
                   # Optional, but if implemented return a short string that will
                   # be printed at the beginning of the break list output for the
                   # breakpoint.

    @param[in] extra_args
       This is an SBStructuredData object that will get passed to the
       constructor of the class in class_name.  You can use this to
       reuse the same class, parametrizing it with entries from this
       dictionary.

    @param module_list
       If this is non-empty, this will be used as the module filter in the
       SearchFilter created for this breakpoint.

    @param file_list
       If this is non-empty, this will be used as the comp unit filter in the
       SearchFilter created for this breakpoint.

    @return
        An SBBreakpoint that will set locations based on the logic in the
        resolver's search callback."
) lldb::SBTarget::BreakpointCreateFromScript;

%feature("docstring", "
    Read breakpoints from source_file and return the newly created
    breakpoints in bkpt_list.

    @param[in] source_file
       The file from which to read the breakpoints

    @param[out] bkpt_list
       A list of the newly created breakpoints.

    @return
        An SBError detailing any errors in reading in the breakpoints."
) lldb::SBTarget::BreakpointsCreateFromFile;

%feature("docstring", "
    Read breakpoints from source_file and return the newly created
    breakpoints in bkpt_list.

    @param[in] source_file
       The file from which to read the breakpoints

    @param[in] matching_names
       Only read in breakpoints whose names match one of the names in this
       list.

    @param[out] bkpt_list
       A list of the newly created breakpoints.

    @return
        An SBError detailing any errors in reading in the breakpoints."
) lldb::SBTarget::BreakpointsCreateFromFile;

%feature("docstring", "
    Write breakpoints to dest_file.

    @param[in] dest_file
       The file to which to write the breakpoints.

    @return
        An SBError detailing any errors in writing in the breakpoints."
) lldb::SBTarget::BreakkpointsWriteToFile;

%feature("docstring", "
    Write breakpoints listed in bkpt_list to dest_file.

    @param[in] dest_file
       The file to which to write the breakpoints.

    @param[in] bkpt_list
       Only write breakpoints from this list.

    @param[in] append
       If true, append the breakpoints in bkpt_list to the others
       serialized in dest_file.  If dest_file doesn't exist, then a new
       file will be created and the breakpoints in bkpt_list written to it.

    @return
        An SBError detailing any errors in writing in the breakpoints."
) lldb::SBTarget::BreakpointsWriteToFile;

%feature("docstring", "
    Create an SBValue with the given name by treating the memory starting at addr as an entity of type.

    @param[in] name
        The name of the resultant SBValue

    @param[in] addr
        The address of the start of the memory region to be used.

    @param[in] type
        The type to use to interpret the memory starting at addr.

    @return
        An SBValue of the given type, may be invalid if there was an error reading
        the underlying memory."
) lldb::SBTarget::CreateValueFromAddress;

%feature("docstring", "
    Disassemble a specified number of instructions starting at an address.

    :param base_addr: the address to start disassembly from.
    :param count: the number of instructions to disassemble.
    :param flavor_string: may be 'intel' or 'att' on x86 targets to specify that style of disassembly.
    :rtype: SBInstructionList
    "
) lldb::SBTarget::ReadInstructions;

%feature("docstring", "
    Disassemble the bytes in a buffer and return them in an SBInstructionList.

    :param base_addr: used for symbolicating the offsets in the byte stream when disassembling.
    :param buf: bytes to be disassembled.
    :param size: (C++) size of the buffer.
    :rtype: SBInstructionList
    "
) lldb::SBTarget::GetInstructions;

%feature("docstring", "
    Disassemble the bytes in a buffer and return them in an SBInstructionList, with a supplied flavor.

    :param base_addr: used for symbolicating the offsets in the byte stream when disassembling.
    :param flavor:  may be 'intel' or 'att' on x86 targets to specify that style of disassembly.
    :param buf: bytes to be disassembled.
    :param size: (C++) size of the buffer.
    :rtype: SBInstructionList
    "
) lldb::SBTarget::GetInstructionsWithFlavor;

%feature("docstring", "
    Returns true if the module has been loaded in this `SBTarget`.
    A module can be loaded either by the dynamic loader or by being manually
    added to the target (see `SBTarget.AddModule` and the ``target module add`` command).

    :rtype: bool
    "
) lldb::SBTarget::IsLoaded;
