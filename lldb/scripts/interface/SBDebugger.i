//===-- SWIG Interface for SBDebugger ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"SBDebugger is the primordial object that creates SBTargets and provides
access to them.  It also manages the overall debugging experiences.

For example (from example/disasm.py),

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
            if not next or next.rstrip('\n') == 'quit':
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
") SBDebugger;
class SBDebugger
{
public:

    static void
    Initialize();
    
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

    void
    Clear ();

    void
    SetAsync (bool b);
    
    bool 
    GetAsync ();

    void
    SkipLLDBInitFiles (bool b);

    void
    SetInputFileHandle (FILE *f, bool transfer_ownership);

    void
    SetOutputFileHandle (FILE *f, bool transfer_ownership);

    void
    SetErrorFileHandle (FILE *f, bool transfer_ownership);

    FILE *
    GetInputFileHandle ();

    FILE *
    GetOutputFileHandle ();

    FILE *
    GetErrorFileHandle ();

    lldb::SBCommandInterpreter
    GetCommandInterpreter ();

    void
    HandleCommand (const char *command);

    lldb::SBListener
    GetListener ();

    void
    HandleProcessEvent (const lldb::SBProcess &process,
                        const lldb::SBEvent &event,
                        FILE *out,
                        FILE *err);

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

    void
    RunCommandInterpreter (bool auto_handle_events,
                           bool spawn_thread,
                           SBCommandInterpreterRunOptions &options,
                           int  &num_errors,
                           bool &quit_requested,
                           bool &stopped_for_crash);
    
    lldb::SBError
    RunREPL (lldb::LanguageType language, const char *repl_options);
}; // class SBDebugger

} // namespace lldb
