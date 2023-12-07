%feature("docstring",
"Represents an executable image and its associated object and symbol files.

The module is designed to be able to select a single slice of an
executable image as it would appear on disk and during program
execution.

You can retrieve SBModule from :py:class:`SBSymbolContext` , which in turn is available
from SBFrame.

SBModule supports symbol iteration, for example, ::

    for symbol in module:
        name = symbol.GetName()
        saddr = symbol.GetStartAddress()
        eaddr = symbol.GetEndAddress()

and rich comparison methods which allow the API program to use, ::

    if thisModule == thatModule:
        print('This module is the same as that module')

to test module equality.  A module also contains object file sections, namely
:py:class:`SBSection` .  SBModule supports section iteration through section_iter(), for
example, ::

    print('Number of sections: %d' % module.GetNumSections())
    for sec in module.section_iter():
        print(sec)

And to iterate the symbols within a SBSection, use symbol_in_section_iter(), ::

    # Iterates the text section and prints each symbols within each sub-section.
    for subsec in text_sec:
        print(INDENT + repr(subsec))
        for sym in exe_module.symbol_in_section_iter(subsec):
            print(INDENT2 + repr(sym))
            print(INDENT2 + 'symbol type: %s' % symbol_type_to_str(sym.GetType()))

produces this following output: ::

    [0x0000000100001780-0x0000000100001d5c) a.out.__TEXT.__text
        id = {0x00000004}, name = 'mask_access(MaskAction, unsigned int)', range = [0x00000001000017c0-0x0000000100001870)
        symbol type: code
        id = {0x00000008}, name = 'thread_func(void*)', range = [0x0000000100001870-0x00000001000019b0)
        symbol type: code
        id = {0x0000000c}, name = 'main', range = [0x00000001000019b0-0x0000000100001d5c)
        symbol type: code
        id = {0x00000023}, name = 'start', address = 0x0000000100001780
        symbol type: code
    [0x0000000100001d5c-0x0000000100001da4) a.out.__TEXT.__stubs
        id = {0x00000024}, name = '__stack_chk_fail', range = [0x0000000100001d5c-0x0000000100001d62)
        symbol type: trampoline
        id = {0x00000028}, name = 'exit', range = [0x0000000100001d62-0x0000000100001d68)
        symbol type: trampoline
        id = {0x00000029}, name = 'fflush', range = [0x0000000100001d68-0x0000000100001d6e)
        symbol type: trampoline
        id = {0x0000002a}, name = 'fgets', range = [0x0000000100001d6e-0x0000000100001d74)
        symbol type: trampoline
        id = {0x0000002b}, name = 'printf', range = [0x0000000100001d74-0x0000000100001d7a)
        symbol type: trampoline
        id = {0x0000002c}, name = 'pthread_create', range = [0x0000000100001d7a-0x0000000100001d80)
        symbol type: trampoline
        id = {0x0000002d}, name = 'pthread_join', range = [0x0000000100001d80-0x0000000100001d86)
        symbol type: trampoline
        id = {0x0000002e}, name = 'pthread_mutex_lock', range = [0x0000000100001d86-0x0000000100001d8c)
        symbol type: trampoline
        id = {0x0000002f}, name = 'pthread_mutex_unlock', range = [0x0000000100001d8c-0x0000000100001d92)
        symbol type: trampoline
        id = {0x00000030}, name = 'rand', range = [0x0000000100001d92-0x0000000100001d98)
        symbol type: trampoline
        id = {0x00000031}, name = 'strtoul', range = [0x0000000100001d98-0x0000000100001d9e)
        symbol type: trampoline
        id = {0x00000032}, name = 'usleep', range = [0x0000000100001d9e-0x0000000100001da4)
        symbol type: trampoline
    [0x0000000100001da4-0x0000000100001e2c) a.out.__TEXT.__stub_helper
    [0x0000000100001e2c-0x0000000100001f10) a.out.__TEXT.__cstring
    [0x0000000100001f10-0x0000000100001f68) a.out.__TEXT.__unwind_info
    [0x0000000100001f68-0x0000000100001ff8) a.out.__TEXT.__eh_frame
"
) lldb::SBModule;

%feature("docstring", "
    Check if the module is file backed.

    @return

        True, if the module is backed by an object file on disk.
        False, if the module is backed by an object file in memory."
) lldb::SBModule::IsFileBacked;

%feature("docstring", "
    Get const accessor for the module file specification.

    This function returns the file for the module on the host system
    that is running LLDB. This can differ from the path on the
    platform since we might be doing remote debugging.

    @return
        A const reference to the file specification object."
) lldb::SBModule::GetFileSpec;

%feature("docstring", "
    Get accessor for the module platform file specification.

    Platform file refers to the path of the module as it is known on
    the remote system on which it is being debugged. For local
    debugging this is always the same as Module::GetFileSpec(). But
    remote debugging might mention a file '/usr/lib/liba.dylib'
    which might be locally downloaded and cached. In this case the
    platform file could be something like:
    '/tmp/lldb/platform-cache/remote.host.computer/usr/lib/liba.dylib'
    The file could also be cached in a local developer kit directory.

    @return
        A const reference to the file specification object."
) lldb::SBModule::GetPlatformFileSpec;

%feature("docstring", "Returns the UUID of the module as a Python string."
) lldb::SBModule::GetUUIDString;

%feature("docstring", "
    Find compile units related to this module and passed source
    file.

    @param[in] sb_file_spec
        A :py:class:`SBFileSpec` object that contains source file
        specification.

    @return
        A :py:class:`SBSymbolContextList` that gets filled in with all of
        the symbol contexts for all the matches."
) lldb::SBModule::FindCompileUnits;

%feature("docstring", "
    Find functions by name.

    @param[in] name
        The name of the function we are looking for.

    @param[in] name_type_mask
        A logical OR of one or more FunctionNameType enum bits that
        indicate what kind of names should be used when doing the
        lookup. Bits include fully qualified names, base names,
        C++ methods, or ObjC selectors.
        See FunctionNameType for more details.

    @return
        A symbol context list that gets filled in with all of the
        matches."
) lldb::SBModule::FindFunctions;

%feature("docstring", "
    Get all types matching type_mask from debug info in this
    module.

    @param[in] type_mask
        A bitfield that consists of one or more bits logically OR'ed
        together from the lldb::TypeClass enumeration. This allows
        you to request only structure types, or only class, struct
        and union types. Passing in lldb::eTypeClassAny will return
        all types found in the debug information for this module.

    @return
        A list of types in this module that match type_mask"
) lldb::SBModule::GetTypes;

%feature("docstring", "
    Find global and static variables by name.

    @param[in] target
        A valid SBTarget instance representing the debuggee.

    @param[in] name
        The name of the global or static variable we are looking
        for.

    @param[in] max_matches
        Allow the number of matches to be limited to max_matches.

    @return
        A list of matched variables in an SBValueList."
) lldb::SBModule::FindGlobalVariables;

%feature("docstring", "
    Find the first global (or static) variable by name.

    @param[in] target
        A valid SBTarget instance representing the debuggee.

    @param[in] name
        The name of the global or static variable we are looking
        for.

    @return
        An SBValue that gets filled in with the found variable (if any)."
) lldb::SBModule::FindFirstGlobalVariable;

%feature("docstring", "
    Returns the number of modules in the module cache. This is an
    implementation detail exposed for testing and should not be relied upon.

    @return
        The number of modules in the module cache."
) lldb::SBModule::GetNumberAllocatedModules;

%feature("docstring", "
    Removes all modules which are no longer needed by any part of LLDB from
    the module cache.

    This is an implementation detail exposed for testing and should not be
    relied upon. Use SBDebugger::MemoryPressureDetected instead to reduce
    LLDB's memory consumption during execution.
") lldb::SBModule::GarbageCollectAllocatedModules;
