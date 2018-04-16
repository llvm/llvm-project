//===-- SWIG Interface for SBTarget -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {


%feature("docstring",
"Represents the target program running under the debugger.

SBTarget supports module, breakpoint, and watchpoint iterations. For example,

    for m in target.module_iter():
        print m

produces:

(x86_64) /Volumes/data/lldb/svn/trunk/test/python_api/lldbutil/iter/a.out
(x86_64) /usr/lib/dyld
(x86_64) /usr/lib/libstdc++.6.dylib
(x86_64) /usr/lib/libSystem.B.dylib
(x86_64) /usr/lib/system/libmathCommon.A.dylib
(x86_64) /usr/lib/libSystem.B.dylib(__commpage)

and,

    for b in target.breakpoint_iter():
        print b

produces:

SBBreakpoint: id = 1, file ='main.cpp', line = 66, locations = 1
SBBreakpoint: id = 2, file ='main.cpp', line = 85, locations = 1

and,

    for wp_loc in target.watchpoint_iter():
        print wp_loc

produces:

Watchpoint 1: addr = 0x1034ca048 size = 4 state = enabled type = rw
    declare @ '/Volumes/data/lldb/svn/trunk/test/python_api/watchpoint/main.c:12'
    hw_index = 0  hit_count = 2     ignore_count = 0"
) SBTarget;
class SBTarget
{
public:
    //------------------------------------------------------------------
    // Broadcaster bits.
    //------------------------------------------------------------------
    enum
    {
        eBroadcastBitBreakpointChanged  = (1 << 0),
        eBroadcastBitModulesLoaded      = (1 << 1),
        eBroadcastBitModulesUnloaded    = (1 << 2),
        eBroadcastBitWatchpointChanged  = (1 << 3),
        eBroadcastBitSymbolsLoaded      = (1 << 4)
    };

    //------------------------------------------------------------------
    // Constructors
    //------------------------------------------------------------------
    SBTarget ();

    SBTarget (const lldb::SBTarget& rhs);

    //------------------------------------------------------------------
    // Destructor
    //------------------------------------------------------------------
    ~SBTarget();

    static const char *
    GetBroadcasterClassName ();
    
    bool
    IsValid() const;

    static bool
    EventIsTargetEvent (const lldb::SBEvent &event);

    static lldb::SBTarget
    GetTargetFromEvent (const lldb::SBEvent &event);

    static uint32_t
    GetNumModulesFromEvent (const lldb::SBEvent &event);

    static lldb::SBModule
    GetModuleAtIndexFromEvent (const uint32_t idx, const lldb::SBEvent &event);

    lldb::SBProcess
    GetProcess ();


    %feature("docstring", "
    //------------------------------------------------------------------
    /// Return the platform object associated with the target.
    ///
    /// After return, the platform object should be checked for
    /// validity.
    ///
    /// @return
    ///     A platform object.
    //------------------------------------------------------------------
    ") GetPlatform;
    lldb::SBPlatform
    GetPlatform ();

    %feature("docstring", "
    //------------------------------------------------------------------
    /// Install any binaries that need to be installed.
    ///
    /// This function does nothing when debugging on the host system.
    /// When connected to remote platforms, the target's main executable
    /// and any modules that have their install path set will be
    /// installed on the remote platform. If the main executable doesn't
    /// have an install location set, it will be installed in the remote
    /// platform's working directory.
    ///
    /// @return
    ///     An error describing anything that went wrong during
    ///     installation.
    //------------------------------------------------------------------
    ") Install;
    lldb::SBError
    Install();

    %feature("docstring", "
    //------------------------------------------------------------------
    /// Launch a new process.
    ///
    /// Launch a new process by spawning a new process using the
    /// target object's executable module's file as the file to launch.
    /// Arguments are given in \a argv, and the environment variables
    /// are in \a envp. Standard input and output files can be
    /// optionally re-directed to \a stdin_path, \a stdout_path, and
    /// \a stderr_path.
    ///
    /// @param[in] listener
    ///     An optional listener that will receive all process events.
    ///     If \a listener is valid then \a listener will listen to all
    ///     process events. If not valid, then this target's debugger
    ///     (SBTarget::GetDebugger()) will listen to all process events. 
    ///
    /// @param[in] argv
    ///     The argument array.
    ///
    /// @param[in] envp
    ///     The environment array.
    ///
    /// @param[in] launch_flags
    ///     Flags to modify the launch (@see lldb::LaunchFlags)
    ///
    /// @param[in] stdin_path
    ///     The path to use when re-directing the STDIN of the new
    ///     process. If all stdXX_path arguments are NULL, a pseudo
    ///     terminal will be used.
    ///
    /// @param[in] stdout_path
    ///     The path to use when re-directing the STDOUT of the new
    ///     process. If all stdXX_path arguments are NULL, a pseudo
    ///     terminal will be used.
    ///
    /// @param[in] stderr_path
    ///     The path to use when re-directing the STDERR of the new
    ///     process. If all stdXX_path arguments are NULL, a pseudo
    ///     terminal will be used.
    ///
    /// @param[in] working_directory
    ///     The working directory to have the child process run in
    ///
    /// @param[in] launch_flags
    ///     Some launch options specified by logical OR'ing 
    ///     lldb::LaunchFlags enumeration values together.
    ///
    /// @param[in] stop_at_entry
    ///     If false do not stop the inferior at the entry point.
    ///
    /// @param[out]
    ///     An error object. Contains the reason if there is some failure.
    ///
    /// @return
    ///      A process object for the newly created process.
    //------------------------------------------------------------------

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
    run to completion if no user interaction is required.
    ") Launch;
    lldb::SBProcess
    Launch (SBListener &listener, 
            char const **argv,
            char const **envp,
            const char *stdin_path,
            const char *stdout_path,
            const char *stderr_path,
            const char *working_directory,
            uint32_t launch_flags,   // See LaunchFlags
            bool stop_at_entry,
            lldb::SBError& error);
            
    %feature("docstring", "
    //------------------------------------------------------------------
    /// Launch a new process with sensible defaults.
    ///
    /// @param[in] argv
    ///     The argument array.
    ///
    /// @param[in] envp
    ///     The environment array.
    ///
    /// @param[in] working_directory
    ///     The working directory to have the child process run in
    ///
    /// Default: listener
    ///     Set to the target's debugger (SBTarget::GetDebugger())
    ///
    /// Default: launch_flags
    ///     Empty launch flags
    ///
    /// Default: stdin_path
    /// Default: stdout_path
    /// Default: stderr_path
    ///     A pseudo terminal will be used.
    ///
    /// @return
    ///      A process object for the newly created process.
    //------------------------------------------------------------------

    For example,

        process = target.LaunchSimple(['X', 'Y', 'Z'], None, os.getcwd())

    launches a new process by passing 'X', 'Y', 'Z' as the args to the
    executable.
    ") LaunchSimple;
    lldb::SBProcess
    LaunchSimple (const char **argv, 
                  const char **envp,
                  const char *working_directory);
    
    lldb::SBProcess
    Launch (lldb::SBLaunchInfo &launch_info, lldb::SBError& error);

    %feature("docstring", "
    //------------------------------------------------------------------
    /// Load a core file
    ///
    /// @param[in] core_file
    ///     File path of the core dump.
    ///
    /// @return
    ///      A process object for the newly created core file.
    //------------------------------------------------------------------

    For example,

        process = target.LoadCore('./a.out.core')

    loads a new core file and returns the process object.
    ") LoadCore;
    lldb::SBProcess
    LoadCore(const char *core_file);
    
    lldb::SBProcess
    Attach (lldb::SBAttachInfo &attach_info, lldb::SBError& error);
    

    %feature("docstring", "
    //------------------------------------------------------------------
    /// Attach to process with pid.
    ///
    /// @param[in] listener
    ///     An optional listener that will receive all process events.
    ///     If \a listener is valid then \a listener will listen to all
    ///     process events. If not valid, then this target's debugger
    ///     (SBTarget::GetDebugger()) will listen to all process events.
    ///
    /// @param[in] pid
    ///     The process ID to attach to.
    ///
    /// @param[out]
    ///     An error explaining what went wrong if attach fails.
    ///
    /// @return
    ///      A process object for the attached process.
    //------------------------------------------------------------------
    ") AttachToProcessWithID;
    lldb::SBProcess
    AttachToProcessWithID (SBListener &listener,
                           lldb::pid_t pid,
                           lldb::SBError& error);

    %feature("docstring", "
    //------------------------------------------------------------------
    /// Attach to process with name.
    ///
    /// @param[in] listener
    ///     An optional listener that will receive all process events.
    ///     If \a listener is valid then \a listener will listen to all
    ///     process events. If not valid, then this target's debugger
    ///     (SBTarget::GetDebugger()) will listen to all process events.
    ///
    /// @param[in] name
    ///     Basename of process to attach to.
    ///
    /// @param[in] wait_for
    ///     If true wait for a new instance of 'name' to be launched.
    ///
    /// @param[out]
    ///     An error explaining what went wrong if attach fails.
    ///
    /// @return
    ///      A process object for the attached process.
    //------------------------------------------------------------------
    ") AttachToProcessWithName;
    lldb::SBProcess
    AttachToProcessWithName (SBListener &listener,
                             const char *name,
                             bool wait_for,
                             lldb::SBError& error);

    %feature("docstring", "
    //------------------------------------------------------------------
    /// Connect to a remote debug server with url.
    ///
    /// @param[in] listener
    ///     An optional listener that will receive all process events.
    ///     If \a listener is valid then \a listener will listen to all
    ///     process events. If not valid, then this target's debugger
    ///     (SBTarget::GetDebugger()) will listen to all process events.
    ///
    /// @param[in] url
    ///     The url to connect to, e.g., 'connect://localhost:12345'.
    ///
    /// @param[in] plugin_name
    ///     The plugin name to be used; can be NULL.
    ///
    /// @param[out]
    ///     An error explaining what went wrong if the connect fails.
    ///
    /// @return
    ///      A process object for the connected process.
    //------------------------------------------------------------------
    ") ConnectRemote;
    lldb::SBProcess
    ConnectRemote (SBListener &listener,
                   const char *url,
                   const char *plugin_name,
                   SBError& error);
    
    lldb::SBFileSpec
    GetExecutable ();

    bool
    AddModule (lldb::SBModule &module);

    lldb::SBModule
    AddModule (const char *path,
               const char *triple,
               const char *uuid);

    lldb::SBModule
    AddModule (const char *path,
               const char *triple,
               const char *uuid_cstr,
               const char *symfile);

    lldb::SBModule
    AddModule (const SBModuleSpec &module_spec);

    uint32_t
    GetNumModules () const;

    lldb::SBModule
    GetModuleAtIndex (uint32_t idx);

    bool
    RemoveModule (lldb::SBModule module);

    lldb::SBDebugger
    GetDebugger() const;

    lldb::SBModule
    FindModule (const lldb::SBFileSpec &file_spec);

    lldb::ByteOrder
    GetByteOrder ();
    
    uint32_t
    GetAddressByteSize();
    
    const char *
    GetTriple ();

    %feature("docstring", "
    //------------------------------------------------------------------
    /// Architecture data byte width accessor
    ///
    /// @return
    /// The size in 8-bit (host) bytes of a minimum addressable
    /// unit from the Architecture's data bus
    //------------------------------------------------------------------
    ") GetDataByteSize;
    uint32_t
    GetDataByteSize ();

    %feature("docstring", "
    //------------------------------------------------------------------
    /// Architecture code byte width accessor
    ///
    /// @return
    /// The size in 8-bit (host) bytes of a minimum addressable
    /// unit from the Architecture's code bus
    //------------------------------------------------------------------
    ") GetCodeByteSize;
    uint32_t
    GetCodeByteSize ();

    lldb::SBError
    SetSectionLoadAddress (lldb::SBSection section,
                           lldb::addr_t section_base_addr);

    lldb::SBError
    ClearSectionLoadAddress (lldb::SBSection section);

    lldb::SBError
    SetModuleLoadAddress (lldb::SBModule module,
                          int64_t sections_offset);

    lldb::SBError
    ClearModuleLoadAddress (lldb::SBModule module);

    %feature("docstring", "
    //------------------------------------------------------------------
    /// Find functions by name.
    ///
    /// @param[in] name
    ///     The name of the function we are looking for.
    ///
    /// @param[in] name_type_mask
    ///     A logical OR of one or more FunctionNameType enum bits that
    ///     indicate what kind of names should be used when doing the
    ///     lookup. Bits include fully qualified names, base names,
    ///     C++ methods, or ObjC selectors. 
    ///     See FunctionNameType for more details.
    ///
    /// @return
    ///     A lldb::SBSymbolContextList that gets filled in with all of 
    ///     the symbol contexts for all the matches.
    //------------------------------------------------------------------
    ") FindFunctions;
    lldb::SBSymbolContextList
    FindFunctions (const char *name, 
                   uint32_t name_type_mask = lldb::eFunctionNameTypeAny);
    
    lldb::SBType
    FindFirstType (const char* type);
    
    lldb::SBTypeList
    FindTypes (const char* type);

    lldb::SBType
    GetBasicType(lldb::BasicType type);

    lldb::SBSourceManager
    GetSourceManager ();

    %feature("docstring", "
    //------------------------------------------------------------------
    /// Find global and static variables by name.
    ///
    /// @param[in] name
    ///     The name of the global or static variable we are looking
    ///     for.
    ///
    /// @param[in] max_matches
    ///     Allow the number of matches to be limited to \a max_matches.
    ///
    /// @return
    ///     A list of matched variables in an SBValueList.
    //------------------------------------------------------------------
    ") FindGlobalVariables;
    lldb::SBValueList
    FindGlobalVariables (const char *name, 
                         uint32_t max_matches);

     %feature("docstring", "
    //------------------------------------------------------------------
    /// Find the first global (or static) variable by name.
    ///
    /// @param[in] name
    ///     The name of the global or static variable we are looking
    ///     for.
    ///
    /// @return
    ///     An SBValue that gets filled in with the found variable (if any).
    //------------------------------------------------------------------
    ") FindFirstGlobalVariable;
    lldb::SBValue
    FindFirstGlobalVariable (const char* name);

    
    lldb::SBValueList
    FindGlobalVariables(const char *name,
                        uint32_t max_matches,
                        MatchType matchtype);

    lldb::SBSymbolContextList
    FindGlobalFunctions(const char *name,
                        uint32_t max_matches,
                        MatchType matchtype);

    void
    Clear ();

     %feature("docstring", "
    //------------------------------------------------------------------
    /// Resolve a current file address into a section offset address.
    ///
    /// @param[in] file_addr
    ///
    /// @return
    ///     An SBAddress which will be valid if...
    //------------------------------------------------------------------
    ") ResolveFileAddress;
    lldb::SBAddress
    ResolveFileAddress (lldb::addr_t file_addr);

    lldb::SBAddress
    ResolveLoadAddress (lldb::addr_t vm_addr);
              
    lldb::SBAddress
    ResolvePastLoadAddress (uint32_t stop_id, lldb::addr_t vm_addr);

    SBSymbolContext
    ResolveSymbolContextForAddress (const SBAddress& addr, 
                                    uint32_t resolve_scope);

     %feature("docstring", "
    //------------------------------------------------------------------
    /// Read target memory. If a target process is running then memory  
    /// is read from here. Otherwise the memory is read from the object
    /// files. For a target whose bytes are sized as a multiple of host
    /// bytes, the data read back will preserve the target's byte order.
    ///
    /// @param[in] addr
    ///     A target address to read from. 
    ///
    /// @param[out] buf
    ///     The buffer to read memory into. 
    ///
    /// @param[in] size
    ///     The maximum number of host bytes to read in the buffer passed
    ///     into this call
    ///
    /// @param[out] error
    ///     Error information is written here if the memory read fails.
    ///
    /// @return
    ///     The amount of data read in host bytes.
    //------------------------------------------------------------------
    ") ReadMemory;
    size_t
    ReadMemory (const SBAddress addr, void *buf, size_t size, lldb::SBError &error);

    lldb::SBBreakpoint
    BreakpointCreateByLocation (const char *file, uint32_t line);

    lldb::SBBreakpoint
    BreakpointCreateByLocation (const lldb::SBFileSpec &file_spec, uint32_t line);

    lldb::SBBreakpoint
    BreakpointCreateByLocation (const lldb::SBFileSpec &file_spec, uint32_t line, lldb::addr_t offset);

    lldb::SBBreakpoint
    BreakpointCreateByLocation (const lldb::SBFileSpec &file_spec, uint32_t line, 
                                lldb::addr_t offset, SBFileSpecList &module_list);

    lldb::SBBreakpoint
    BreakpointCreateByName (const char *symbol_name, const char *module_name = NULL);

    lldb::SBBreakpoint
    BreakpointCreateByName (const char *symbol_name,
                            uint32_t func_name_type,           // Logical OR one or more FunctionNameType enum bits
                            const SBFileSpecList &module_list, 
                            const SBFileSpecList &comp_unit_list);

    lldb::SBBreakpoint
    BreakpointCreateByName (const char *symbol_name,
                            uint32_t func_name_type,           // Logical OR one or more FunctionNameType enum bits
                            lldb::LanguageType symbol_language,
                            const SBFileSpecList &module_list, 
                            const SBFileSpecList &comp_unit_list);

%typemap(in) (const char **symbol_name, uint32_t num_names) {
  using namespace lldb_private;
  /* Check if is a list  */
  if (PythonList::Check($input)) {
    PythonList list(PyRefType::Borrowed, $input);
    $2 = list.GetSize();
    int i = 0;
    $1 = (char**)malloc(($2+1)*sizeof(char*));
    for (i = 0; i < $2; i++) {
      PythonString py_str = list.GetItemAtIndex(i).AsType<PythonString>();
      if (!py_str.IsAllocated()) {
        PyErr_SetString(PyExc_TypeError,"list must contain strings and blubby");
        free($1);
        return nullptr;
      }

      $1[i] = const_cast<char*>(py_str.GetString().data());
    }
    $1[i] = 0;
  } else if ($input == Py_None) {
    $1 =  NULL;
  } else {
    PyErr_SetString(PyExc_TypeError,"not a list");
    return NULL;
  }
}

//%typecheck(SWIG_TYPECHECK_STRING_ARRAY) (const char *symbol_name[], uint32_t num_names) {
//    $1 = 1;
//    $2 = 1;
//}

    lldb::SBBreakpoint
    BreakpointCreateByNames (const char **symbol_name,
                             uint32_t num_names,
                             uint32_t name_type_mask,           // Logical OR one or more FunctionNameType enum bits
                             const SBFileSpecList &module_list,
                             const SBFileSpecList &comp_unit_list);

    lldb::SBBreakpoint
    BreakpointCreateByNames (const char **symbol_name,
                             uint32_t num_names,
                             uint32_t name_type_mask,           // Logical OR one or more FunctionNameType enum bits
                             lldb::LanguageType symbol_language,
                             const SBFileSpecList &module_list,
                             const SBFileSpecList &comp_unit_list);

    lldb::SBBreakpoint
    BreakpointCreateByNames (const char **symbol_name,
                             uint32_t num_names,
                             uint32_t name_type_mask,           // Logical OR one or more FunctionNameType enum bits
                             lldb::LanguageType symbol_language,
                             lldb::addr_t offset,
                             const SBFileSpecList &module_list,
                             const SBFileSpecList &comp_unit_list);

    lldb::SBBreakpoint
    BreakpointCreateByRegex (const char *symbol_name_regex, const char *module_name = NULL);

    lldb::SBBreakpoint
    BreakpointCreateByRegex (const char *symbol_name_regex,
                             lldb::LanguageType symbol_language,
                             const SBFileSpecList &module_list, 
                             const SBFileSpecList &comp_unit_list);

    lldb::SBBreakpoint
    BreakpointCreateBySourceRegex (const char *source_regex, const lldb::SBFileSpec &source_file, const char *module_name = NULL);

    lldb::SBBreakpoint
    BreakpointCreateBySourceRegex (const char *source_regex, const lldb::SBFileSpecList &module_list, const lldb::SBFileSpecList &file_list);

    lldb::SBBreakpoint
    BreakpointCreateBySourceRegex (const char *source_regex,
                                   const SBFileSpecList &module_list,
                                   const SBFileSpecList &source_file,
                                   const SBStringList  &func_names);

    lldb::SBBreakpoint
    BreakpointCreateForException  (lldb::LanguageType language,
                                   bool catch_bp,
                                   bool throw_bp);

    lldb::SBBreakpoint
    BreakpointCreateByAddress (addr_t address);

    lldb::SBBreakpoint
    BreakpointCreateBySBAddress (SBAddress &sb_address);

    uint32_t
    GetNumBreakpoints () const;

    lldb::SBBreakpoint
    GetBreakpointAtIndex (uint32_t idx) const;

    bool
    BreakpointDelete (break_id_t break_id);

    lldb::SBBreakpoint
    FindBreakpointByID (break_id_t break_id);

  
    bool FindBreakpointsByName(const char *name, SBBreakpointList &bkpt_list);

    void DeleteBreakpointName(const char *name);

    void GetBreakpointNames(SBStringList &names);

    bool
    EnableAllBreakpoints ();

    bool
    DisableAllBreakpoints ();

    bool
    DeleteAllBreakpoints ();

     %feature("docstring", "
    //------------------------------------------------------------------
    /// Read breakpoints from source_file and return the newly created 
    /// breakpoints in bkpt_list.
    ///
    /// @param[in] source_file
    ///    The file from which to read the breakpoints
    /// 
    /// @param[out] bkpt_list
    ///    A list of the newly created breakpoints.
    ///
    /// @return
    ///     An SBError detailing any errors in reading in the breakpoints.
    //------------------------------------------------------------------
    ") BreakpointsCreateFromFile;
    lldb::SBError
    BreakpointsCreateFromFile(SBFileSpec &source_file, 
                              SBBreakpointList &bkpt_list);

     %feature("docstring", "
    //------------------------------------------------------------------
    /// Read breakpoints from source_file and return the newly created 
    /// breakpoints in bkpt_list.
    ///
    /// @param[in] source_file
    ///    The file from which to read the breakpoints
    ///
    /// @param[in] matching_names
    ///    Only read in breakpoints whose names match one of the names in this
    ///    list.
    /// 
    /// @param[out] bkpt_list
    ///    A list of the newly created breakpoints.
    ///
    /// @return
    ///     An SBError detailing any errors in reading in the breakpoints.
    //------------------------------------------------------------------
    ") BreakpointsCreateFromFile;
    lldb::SBError BreakpointsCreateFromFile(SBFileSpec &source_file,
                                          SBStringList &matching_names,
                                          SBBreakpointList &new_bps);

     %feature("docstring", "
    //------------------------------------------------------------------
    /// Write breakpoints to dest_file.
    ///
    /// @param[in] dest_file
    ///    The file to which to write the breakpoints.
    ///
    /// @return
    ///     An SBError detailing any errors in writing in the breakpoints.
    //------------------------------------------------------------------
    ") BreakpointsCreateFromFile;
    lldb::SBError
    BreakpointsWriteToFile(SBFileSpec &dest_file);
      
     %feature("docstring", "
    //------------------------------------------------------------------
    /// Write breakpoints listed in bkpt_list to dest_file.
    ///
    /// @param[in] dest_file
    ///    The file to which to write the breakpoints.
    ///
    /// @param[in] bkpt_list
    ///    Only write breakpoints from this list.
    ///
    /// @param[in] append
    ///    If \btrue, append the breakpoints in bkpt_list to the others
    ///    serialized in dest_file.  If dest_file doesn't exist, then a new
    ///    file will be created and the breakpoints in bkpt_list written to it.
    ///
    /// @return
    ///     An SBError detailing any errors in writing in the breakpoints.
    //------------------------------------------------------------------
    ") BreakpointsCreateFromFile;
    lldb::SBError
    BreakpointsWriteToFile(SBFileSpec &dest_file, 
                           SBBreakpointList &bkpt_list,
                           bool append = false);

    uint32_t
    GetNumWatchpoints () const;
    
    lldb::SBWatchpoint
    GetWatchpointAtIndex (uint32_t idx) const;
    
    bool
    DeleteWatchpoint (lldb::watch_id_t watch_id);
    
    lldb::SBWatchpoint
    FindWatchpointByID (lldb::watch_id_t watch_id);
    
    bool
    EnableAllWatchpoints ();
    
    bool
    DisableAllWatchpoints ();
    
    bool
    DeleteAllWatchpoints ();

    lldb::SBWatchpoint
    WatchAddress (lldb::addr_t addr, 
                  size_t size, 
                  bool read, 
                  bool write,
                  SBError &error);
             

    lldb::SBBroadcaster
    GetBroadcaster () const;
              
     %feature("docstring", "
    //------------------------------------------------------------------
    /// Create an SBValue with the given name by treating the memory starting at addr as an entity of type.
    ///
    /// @param[in] name
    ///     The name of the resultant SBValue
    ///
    /// @param[in] addr
    ///     The address of the start of the memory region to be used.
    ///
    /// @param[in] type
    ///     The type to use to interpret the memory starting at addr.
    ///
    /// @return
    ///     An SBValue of the given type, may be invalid if there was an error reading
    ///     the underlying memory.
    //------------------------------------------------------------------
    ") CreateValueFromAddress;
    lldb::SBValue
    CreateValueFromAddress (const char *name, lldb::SBAddress addr, lldb::SBType type);

    lldb::SBValue
    CreateValueFromData (const char *name, lldb::SBData data, lldb::SBType type);
  
    lldb::SBValue
    CreateValueFromExpression (const char *name, const char* expr);
              
    %feature("docstring", "
    Disassemble a specified number of instructions starting at an address.
    Parameters:
       base_addr       -- the address to start disassembly from
       count           -- the number of instructions to disassemble
       flavor_string   -- may be 'intel' or 'att' on x86 targets to specify that style of disassembly
    Returns an SBInstructionList.") 
    ReadInstructions;
    lldb::SBInstructionList
    ReadInstructions (lldb::SBAddress base_addr, uint32_t count);    

    lldb::SBInstructionList
    ReadInstructions (lldb::SBAddress base_addr, uint32_t count, const char *flavor_string);

    %feature("docstring", "
    Disassemble the bytes in a buffer and return them in an SBInstructionList.
    Parameters:
       base_addr -- used for symbolicating the offsets in the byte stream when disassembling
       buf       -- bytes to be disassembled
       size      -- (C++) size of the buffer
    Returns an SBInstructionList.") 
    GetInstructions;
    lldb::SBInstructionList
    GetInstructions (lldb::SBAddress base_addr, const void *buf, size_t size);

    %feature("docstring", "
    Disassemble the bytes in a buffer and return them in an SBInstructionList, with a supplied flavor.
    Parameters:
       base_addr -- used for symbolicating the offsets in the byte stream when disassembling
       flavor    -- may be 'intel' or 'att' on x86 targets to specify that style of disassembly
       buf       -- bytes to be disassembled
       size      -- (C++) size of the buffer
    Returns an SBInstructionList.") 
    GetInstructionsWithFlavor;
    lldb::SBInstructionList
    GetInstructionsWithFlavor (lldb::SBAddress base_addr, const char *flavor_string, const void *buf, size_t size);
    
    lldb::SBSymbolContextList
    FindSymbols (const char *name, lldb::SymbolType type = eSymbolTypeAny);

    bool
    GetDescription (lldb::SBStream &description, lldb::DescriptionLevel description_level);
    
    lldb::addr_t
    GetStackRedZoneSize();

    lldb::SBLaunchInfo
    GetLaunchInfo () const;

    void
    SetLaunchInfo (const lldb::SBLaunchInfo &launch_info);

    lldb::SBStructuredData GetStatistics();

    bool
    operator == (const lldb::SBTarget &rhs) const;

    bool
    operator != (const lldb::SBTarget &rhs) const;

    lldb::SBValue
    EvaluateExpression (const char *expr);

    lldb::SBValue
    EvaluateExpression (const char *expr, const lldb::SBExpressionOptions &options);

    %pythoncode %{
        class modules_access(object):
            '''A helper object that will lazily hand out lldb.SBModule objects for a target when supplied an index, or by full or partial path.'''
            def __init__(self, sbtarget):
                self.sbtarget = sbtarget
        
            def __len__(self):
                if self.sbtarget:
                    return int(self.sbtarget.GetNumModules())
                return 0
        
            def __getitem__(self, key):
                num_modules = self.sbtarget.GetNumModules()
                if type(key) is int:
                    if key < num_modules:
                        return self.sbtarget.GetModuleAtIndex(key)
                elif type(key) is str:
                    if key.find('/') == -1:
                        for idx in range(num_modules):
                            module = self.sbtarget.GetModuleAtIndex(idx)
                            if module.file.basename == key:
                                return module
                    else:
                        for idx in range(num_modules):
                            module = self.sbtarget.GetModuleAtIndex(idx)
                            if module.file.fullpath == key:
                                return module
                    # See if the string is a UUID
                    try:
                        the_uuid = uuid.UUID(key)
                        if the_uuid:
                            for idx in range(num_modules):
                                module = self.sbtarget.GetModuleAtIndex(idx)
                                if module.uuid == the_uuid:
                                    return module
                    except:
                        return None
                elif type(key) is uuid.UUID:
                    for idx in range(num_modules):
                        module = self.sbtarget.GetModuleAtIndex(idx)
                        if module.uuid == key:
                            return module
                elif type(key) is re.SRE_Pattern:
                    matching_modules = []
                    for idx in range(num_modules):
                        module = self.sbtarget.GetModuleAtIndex(idx)
                        re_match = key.search(module.path.fullpath)
                        if re_match:
                            matching_modules.append(module)
                    return matching_modules
                else:
                    print("error: unsupported item type: %s" % type(key))
                return None
        
        def get_modules_access_object(self):
            '''An accessor function that returns a modules_access() object which allows lazy module access from a lldb.SBTarget object.'''
            return self.modules_access (self)
        
        def get_modules_array(self):
            '''An accessor function that returns a list() that contains all modules in a lldb.SBTarget object.'''
            modules = []
            for idx in range(self.GetNumModules()):
                modules.append(self.GetModuleAtIndex(idx))
            return modules

        __swig_getmethods__["modules"] = get_modules_array
        if _newclass: modules = property(get_modules_array, None, doc='''A read only property that returns a list() of lldb.SBModule objects contained in this target. This list is a list all modules that the target currently is tracking (the main executable and all dependent shared libraries).''')

        __swig_getmethods__["module"] = get_modules_access_object
        if _newclass: module = property(get_modules_access_object, None, doc=r'''A read only property that returns an object that implements python operator overloading with the square brackets().\n    target.module[<int>] allows array access to any modules.\n    target.module[<str>] allows access to modules by basename, full path, or uuid string value.\n    target.module[uuid.UUID()] allows module access by UUID.\n    target.module[re] allows module access using a regular expression that matches the module full path.''')

        __swig_getmethods__["process"] = GetProcess
        if _newclass: process = property(GetProcess, None, doc='''A read only property that returns an lldb object that represents the process (lldb.SBProcess) that this target owns.''')

        __swig_getmethods__["executable"] = GetExecutable
        if _newclass: executable = property(GetExecutable, None, doc='''A read only property that returns an lldb object that represents the main executable module (lldb.SBModule) for this target.''')

        __swig_getmethods__["debugger"] = GetDebugger
        if _newclass: debugger = property(GetDebugger, None, doc='''A read only property that returns an lldb object that represents the debugger (lldb.SBDebugger) that owns this target.''')

        __swig_getmethods__["num_breakpoints"] = GetNumBreakpoints
        if _newclass: num_breakpoints = property(GetNumBreakpoints, None, doc='''A read only property that returns the number of breakpoints that this target has as an integer.''')

        __swig_getmethods__["num_watchpoints"] = GetNumWatchpoints
        if _newclass: num_watchpoints = property(GetNumWatchpoints, None, doc='''A read only property that returns the number of watchpoints that this target has as an integer.''')

        __swig_getmethods__["broadcaster"] = GetBroadcaster
        if _newclass: broadcaster = property(GetBroadcaster, None, doc='''A read only property that an lldb object that represents the broadcaster (lldb.SBBroadcaster) for this target.''')
        
        __swig_getmethods__["byte_order"] = GetByteOrder
        if _newclass: byte_order = property(GetByteOrder, None, doc='''A read only property that returns an lldb enumeration value (lldb.eByteOrderLittle, lldb.eByteOrderBig, lldb.eByteOrderInvalid) that represents the byte order for this target.''')
        
        __swig_getmethods__["addr_size"] = GetAddressByteSize
        if _newclass: addr_size = property(GetAddressByteSize, None, doc='''A read only property that returns the size in bytes of an address for this target.''')
        
        __swig_getmethods__["triple"] = GetTriple
        if _newclass: triple = property(GetTriple, None, doc='''A read only property that returns the target triple (arch-vendor-os) for this target as a string.''')

        __swig_getmethods__["data_byte_size"] = GetDataByteSize
        if _newclass: data_byte_size = property(GetDataByteSize, None, doc='''A read only property that returns the size in host bytes of a byte in the data address space for this target.''')

        __swig_getmethods__["code_byte_size"] = GetCodeByteSize
        if _newclass: code_byte_size = property(GetCodeByteSize, None, doc='''A read only property that returns the size in host bytes of a byte in the code address space for this target.''')

        __swig_getmethods__["platform"] = GetPlatform
        if _newclass: platform = property(GetPlatform, None, doc='''A read only property that returns the platform associated with with this target.''')
    %}

};

} // namespace lldb
