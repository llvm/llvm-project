//===-- SWIG Interface for SBDebugger ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"SBDebugger is the primordial object that creates SBTargets and provides
access to them.  It also manages the overall debugging experiences.

For example (from example/disasm.py),::

    import lldb
    import os
    import sys

    def disassemble_instructions (insts):
        for i in insts:
            print i

    ...

    # Create a new debugger instance
    debugger = lldb.SBDebugger.Create()

    # When we step or continue, don't return from the function until the process
    # stops. We do this by setting the async mode to false.
    debugger.SetAsync (False)

    # Create a target from a file and arch
    print('Creating a target for \'%s\'' % exe)

    target = debugger.CreateTargetWithFileAndArch (exe, lldb.LLDB_ARCH_DEFAULT)

    if target:
        # If the target is valid set a breakpoint at main
        main_bp = target.BreakpointCreateByName (fname, target.GetExecutable().GetFilename());

        print main_bp

        # Launch the process. Since we specified synchronous mode, we won't return
        # from this function until we hit the breakpoint at main
        process = target.LaunchSimple (None, None, os.getcwd())

        # Make sure the launch went ok
        if process:
            # Print some simple process info
            state = process.GetState ()
            print process
            if state == lldb.eStateStopped:
                # Get the first thread
                thread = process.GetThreadAtIndex (0)
                if thread:
                    # Print some simple thread info
                    print thread
                    # Get the first frame
                    frame = thread.GetFrameAtIndex (0)
                    if frame:
                        # Print some simple frame info
                        print frame
                        function = frame.GetFunction()
                        # See if we have debug info (a function)
                        if function:
                            # We do have a function, print some info for the function
                            print function
                            # Now get all instructions for this function and print them
                            insts = function.GetInstructions(target)
                            disassemble_instructions (insts)
                        else:
                            # See if we have a symbol in the symbol table for where we stopped
                            symbol = frame.GetSymbol();
                            if symbol:
                                # We do have a symbol, print some info for the symbol
                                print symbol
                                # Now get all instructions for this symbol and print them
                                insts = symbol.GetInstructions(target)
                                disassemble_instructions (insts)

                        registerList = frame.GetRegisters()
                        print('Frame registers (size of register set = %d):' % registerList.GetSize())
                        for value in registerList:
                            #print value
                            print('%s (number of children = %d):' % (value.GetName(), value.GetNumChildren()))
                            for child in value:
                                print('Name: ', child.GetName(), ' Value: ', child.GetValue())

                print('Hit the breakpoint at main, enter to continue and wait for program to exit or \'Ctrl-D\'/\'quit\' to terminate the program')
                next = sys.stdin.readline()
                if not next or next.rstrip('\\n') == 'quit':
                    print('Terminating the inferior process...')
                    process.Kill()
                else:
                    # Now continue to the program exit
                    process.Continue()
                    # When we return from the above function we will hopefully be at the
                    # program exit. Print out some process info
                    print process
            elif state == lldb.eStateExited:
                print('Didn\'t hit the breakpoint at main, program has exited...')
            else:
                print('Unexpected process state: %s, killing process...' % debugger.StateAsCString (state))
                process.Kill()

Sometimes you need to create an empty target that will get filled in later.  The most common use for this
is to attach to a process by name or pid where you don't know the executable up front.  The most convenient way
to do this is: ::

    target = debugger.CreateTarget('')
    error = lldb.SBError()
    process = target.AttachToProcessWithName(debugger.GetListener(), 'PROCESS_NAME', False, error)

or the equivalent arguments for :py:class:`SBTarget.AttachToProcessWithID` .") SBDebugger;
class SBDebugger
{
public:
    enum
    {
        eBroadcastBitProgress = (1 << 0),
        eBroadcastBitWarning = (1 << 1),
        eBroadcastBitError = (1 << 2),
    };


    static const char *GetProgressFromEvent(const lldb::SBEvent &event,
                                        uint64_t &OUTPUT,
                                        uint64_t &OUTPUT,
                                        uint64_t &OUTPUT,
                                        bool &OUTPUT);

    static lldb::SBStructuredData GetProgressDataFromEvent(const lldb::SBEvent &event);

    static lldb::SBStructuredData GetDiagnosticFromEvent(const lldb::SBEvent &event);

    SBBroadcaster GetBroadcaster();

    static void
    Initialize();

    static SBError
    InitializeWithErrorHandling();

    static void PrintStackTraceOnError();

    static void
    Terminate();

    static lldb::SBDebugger
    Create();

    static lldb::SBDebugger
    Create(bool source_init_files);

    static lldb::SBDebugger
    Create(bool source_init_files, lldb::LogOutputCallback log_callback, void *baton);

    static void
    Destroy (lldb::SBDebugger &debugger);

    static void
    MemoryPressureDetected();

    SBDebugger();

    SBDebugger(const lldb::SBDebugger &rhs);

    ~SBDebugger();

    bool
    IsValid() const;

    explicit operator bool() const;

    void
    Clear ();

    void
    SetAsync (bool b);

    bool
    GetAsync ();

    void
    SkipLLDBInitFiles (bool b);

#ifdef SWIGPYTHON
    %pythoncode %{
        def SetOutputFileHandle(self, file, transfer_ownership):
            "DEPRECATED, use SetOutputFile"
            if file is None:
                import sys
                file = sys.stdout
            self.SetOutputFile(SBFile.Create(file, borrow=True))

        def SetInputFileHandle(self, file, transfer_ownership):
            "DEPRECATED, use SetInputFile"
            if file is None:
                import sys
                file = sys.stdin
            self.SetInputFile(SBFile.Create(file, borrow=True))

        def SetErrorFileHandle(self, file, transfer_ownership):
            "DEPRECATED, use SetErrorFile"
            if file is None:
                import sys
                file = sys.stderr
            self.SetErrorFile(SBFile.Create(file, borrow=True))
    %}
#endif


    %extend {

        lldb::FileSP GetInputFileHandle() {
            return self->GetInputFile().GetFile();
        }

        lldb::FileSP GetOutputFileHandle() {
            return self->GetOutputFile().GetFile();
        }

        lldb::FileSP GetErrorFileHandle() {
            return self->GetErrorFile().GetFile();
        }
    }

    lldb::SBStructuredData GetSetting(const char *setting = nullptr);

    SBError
    SetInputString (const char* data);

    SBError
    SetInputFile (SBFile file);

    SBError
    SetOutputFile (SBFile file);

    SBError
    SetErrorFile (SBFile file);

    SBError
    SetInputFile (FileSP file);

    SBError
    SetOutputFile (FileSP file);

    SBError
    SetErrorFile (FileSP file);

    SBFile
    GetInputFile ();

    SBFile
    GetOutputFile ();

    SBFile
    GetErrorFile ();

    lldb::SBCommandInterpreter
    GetCommandInterpreter ();

    void
    HandleCommand (const char *command);

    lldb::SBListener
    GetListener ();

    void
    HandleProcessEvent (const lldb::SBProcess &process,
                        const lldb::SBEvent &event,
                        SBFile out,
                        SBFile err);

    void
    HandleProcessEvent (const lldb::SBProcess &process,
                        const lldb::SBEvent &event,
                        FileSP BORROWED,
                        FileSP BORROWED);

    lldb::SBTarget
    CreateTarget (const char *filename,
                  const char *target_triple,
                  const char *platform_name,
                  bool add_dependent_modules,
                  lldb::SBError& sb_error);

    lldb::SBTarget
    CreateTargetWithFileAndTargetTriple (const char *filename,
                                         const char *target_triple);

    lldb::SBTarget
    CreateTargetWithFileAndArch (const char *filename,
                                 const char *archname);

    lldb::SBTarget
    CreateTarget (const char *filename);

    %feature("docstring",
    "The dummy target holds breakpoints and breakpoint names that will prime newly created targets."
    ) GetDummyTarget;
    lldb::SBTarget GetDummyTarget();

    %feature("docstring",
    "Return true if target is deleted from the target list of the debugger."
    ) DeleteTarget;
    bool
    DeleteTarget (lldb::SBTarget &target);

    lldb::SBTarget
    GetTargetAtIndex (uint32_t idx);

    uint32_t
    GetIndexOfTarget (lldb::SBTarget target);

    lldb::SBTarget
    FindTargetWithProcessID (pid_t pid);

    lldb::SBTarget
    FindTargetWithFileAndArch (const char *filename,
                               const char *arch);

    uint32_t
    GetNumTargets ();

    lldb::SBTarget
    GetSelectedTarget ();

    void
    SetSelectedTarget (lldb::SBTarget &target);

    lldb::SBPlatform
    GetSelectedPlatform();

    void
    SetSelectedPlatform(lldb::SBPlatform &platform);

    %feature("docstring",
    "Get the number of currently active platforms."
    ) GetNumPlatforms;
    uint32_t
    GetNumPlatforms ();

    %feature("docstring",
    "Get one of the currently active platforms."
    ) GetPlatformAtIndex;
    lldb::SBPlatform
    GetPlatformAtIndex (uint32_t idx);

    %feature("docstring",
    "Get the number of available platforms."
    ) GetNumAvailablePlatforms;
    uint32_t
    GetNumAvailablePlatforms ();

    %feature("docstring", "
    Get the name and description of one of the available platforms.

    @param idx Zero-based index of the platform for which info should be
               retrieved, must be less than the value returned by
               GetNumAvailablePlatforms().") GetAvailablePlatformInfoAtIndex;
    lldb::SBStructuredData
    GetAvailablePlatformInfoAtIndex (uint32_t idx);

    lldb::SBSourceManager
    GetSourceManager ();

    // REMOVE: just for a quick fix, need to expose platforms through
    // SBPlatform from this class.
    lldb::SBError
    SetCurrentPlatform (const char *platform_name);

    bool
    SetCurrentPlatformSDKRoot (const char *sysroot);

    // FIXME: Once we get the set show stuff in place, the driver won't need
    // an interface to the Set/Get UseExternalEditor.
    bool
    SetUseExternalEditor (bool input);

    bool
    GetUseExternalEditor ();

    bool
    SetUseColor (bool use_color);

    bool
    GetUseColor () const;

    static bool
    GetDefaultArchitecture (char *arch_name, size_t arch_name_len);

    static bool
    SetDefaultArchitecture (const char *arch_name);

    lldb::ScriptLanguage
    GetScriptingLanguage (const char *script_language_name);

    static const char *
    GetVersionString ();

    static const char *
    StateAsCString (lldb::StateType state);

    static SBStructuredData GetBuildConfiguration();

    static bool
    StateIsRunningState (lldb::StateType state);

    static bool
    StateIsStoppedState (lldb::StateType state);

    bool
    EnableLog (const char *channel, const char ** types);

    void
    SetLoggingCallback (lldb::LogOutputCallback log_callback, void *baton);

    void
    DispatchInput (const void *data, size_t data_len);

    void
    DispatchInputInterrupt ();

    void
    DispatchInputEndOfFile ();

    const char *
    GetInstanceName  ();

    static SBDebugger
    FindDebuggerWithID (int id);

    static lldb::SBError
    SetInternalVariable (const char *var_name, const char *value, const char *debugger_instance_name);

    static lldb::SBStringList
    GetInternalVariableValue (const char *var_name, const char *debugger_instance_name);

    bool
    GetDescription (lldb::SBStream &description);

    uint32_t
    GetTerminalWidth () const;

    void
    SetTerminalWidth (uint32_t term_width);

    lldb::user_id_t
    GetID ();

    const char *
    GetPrompt() const;

    void
    SetPrompt (const char *prompt);

    const char *
    GetReproducerPath() const;

    lldb::ScriptLanguage
    GetScriptLanguage() const;

    void
    SetScriptLanguage (lldb::ScriptLanguage script_lang);

    bool
    GetCloseInputOnEOF () const;

    void
    SetCloseInputOnEOF (bool b);

    lldb::SBTypeCategory
    GetCategory (const char* category_name);

    SBTypeCategory
    GetCategory (lldb::LanguageType lang_type);

    lldb::SBTypeCategory
    CreateCategory (const char* category_name);

    bool
    DeleteCategory (const char* category_name);

    uint32_t
    GetNumCategories ();

    lldb::SBTypeCategory
    GetCategoryAtIndex (uint32_t);

    lldb::SBTypeCategory
    GetDefaultCategory();

    lldb::SBTypeFormat
    GetFormatForType (lldb::SBTypeNameSpecifier);

    lldb::SBTypeSummary
    GetSummaryForType (lldb::SBTypeNameSpecifier);

    lldb::SBTypeFilter
    GetFilterForType (lldb::SBTypeNameSpecifier);

    lldb::SBTypeSynthetic
    GetSyntheticForType (lldb::SBTypeNameSpecifier);

    SBStructuredData GetScriptInterpreterInfo(ScriptLanguage);

    STRING_EXTENSION(SBDebugger)

    %feature("docstring",
"Launch a command interpreter session. Commands are read from standard input or
from the input handle specified for the debugger object. Output/errors are
similarly redirected to standard output/error or the configured handles.

@param[in] auto_handle_events If true, automatically handle resulting events.
@param[in] spawn_thread If true, start a new thread for IO handling.
@param[in] options Parameter collection of type SBCommandInterpreterRunOptions.
@param[in] num_errors Initial error counter.
@param[in] quit_requested Initial quit request flag.
@param[in] stopped_for_crash Initial crash flag.

@return
A tuple with the number of errors encountered by the interpreter, a boolean
indicating whether quitting the interpreter was requested and another boolean
set to True in case of a crash.

Example: ::

    # Start an interactive lldb session from a script (with a valid debugger object
    # created beforehand):
    n_errors, quit_requested, has_crashed = debugger.RunCommandInterpreter(True,
        False, lldb.SBCommandInterpreterRunOptions(), 0, False, False)") RunCommandInterpreter;
    %apply int& INOUT { int& num_errors };
    %apply bool& INOUT { bool& quit_requested };
    %apply bool& INOUT { bool& stopped_for_crash };
    void
    RunCommandInterpreter (bool auto_handle_events,
                           bool spawn_thread,
                           SBCommandInterpreterRunOptions &options,
                           int  &num_errors,
                           bool &quit_requested,
                           bool &stopped_for_crash);

    lldb::SBError
    RunREPL (lldb::LanguageType language, const char *repl_options);

    SBTrace LoadTraceFromFile(SBError &error, const SBFileSpec &trace_description_file);

#ifdef SWIGPYTHON
    %pythoncode%{
    def __iter__(self):
        '''Iterate over all targets in a lldb.SBDebugger object.'''
        return lldb_iter(self, 'GetNumTargets', 'GetTargetAtIndex')

    def __len__(self):
        '''Return the number of targets in a lldb.SBDebugger object.'''
        return self.GetNumTargets()
    %}
#endif

}; // class SBDebugger

} // namespace lldb
