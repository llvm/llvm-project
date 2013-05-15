//===-- SBTarget.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBTarget_h_
#define LLDB_SBTarget_h_

#include "lldb/API/SBDefines.h"
#include "lldb/API/SBAddress.h"
#include "lldb/API/SBBroadcaster.h"
#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBFileSpecList.h"
#include "lldb/API/SBSymbolContextList.h"
#include "lldb/API/SBType.h"
#include "lldb/API/SBValue.h"
#include "lldb/API/SBWatchpoint.h"

namespace lldb {

class SBLaunchInfo
{
public:
    SBLaunchInfo (const char **argv);
    
    ~SBLaunchInfo();

    uint32_t
    GetUserID();
    
    uint32_t
    GetGroupID();
    
    bool
    UserIDIsValid ();
    
    bool
    GroupIDIsValid ();
    
    void
    SetUserID (uint32_t uid);
    
    void
    SetGroupID (uint32_t gid);
    
    uint32_t
    GetNumArguments ();
    
    const char *
    GetArgumentAtIndex (uint32_t idx);
    
    void
    SetArguments (const char **argv, bool append);
    
    uint32_t
    GetNumEnvironmentEntries ();
    
    const char *
    GetEnvironmentEntryAtIndex (uint32_t idx);
    
    void
    SetEnvironmentEntries (const char **envp, bool append);
    
    void
    Clear ();
    
    const char *
    GetWorkingDirectory () const;
    
    void
    SetWorkingDirectory (const char *working_dir);
    
    uint32_t
    GetLaunchFlags ();
    
    void
    SetLaunchFlags (uint32_t flags);
    
    const char *
    GetProcessPluginName ();
    
    void
    SetProcessPluginName (const char *plugin_name);
    
    const char *
    GetShell ();
    
    void
    SetShell (const char * path);
    
    uint32_t
    GetResumeCount ();
    
    void
    SetResumeCount (uint32_t c);
    
    bool
    AddCloseFileAction (int fd);
    
    bool
    AddDuplicateFileAction (int fd, int dup_fd);
    
    bool
    AddOpenFileAction (int fd, const char *path, bool read, bool write);
    
    bool
    AddSuppressFileAction (int fd, bool read, bool write);
    
protected:
    friend class SBTarget;
    
    lldb_private::ProcessLaunchInfo &
    ref ();

    ProcessLaunchInfoSP m_opaque_sp;
};

class SBAttachInfo
{
public:
    SBAttachInfo ();
    
    SBAttachInfo (lldb::pid_t pid);
    
    SBAttachInfo (const char *path, bool wait_for);
    
    SBAttachInfo (const SBAttachInfo &rhs);
    
    ~SBAttachInfo();

    SBAttachInfo &
    operator = (const SBAttachInfo &rhs);
    
    lldb::pid_t
    GetProcessID ();
    
    void
    SetProcessID (lldb::pid_t pid);
    
    void
    SetExecutable (const char *path);
    
    void
    SetExecutable (lldb::SBFileSpec exe_file);
    
    bool
    GetWaitForLaunch ();
    
    void
    SetWaitForLaunch (bool b);
    
    bool
    GetIgnoreExisting ();
    
    void
    SetIgnoreExisting (bool b);
    
    uint32_t
    GetResumeCount ();
    
    void
    SetResumeCount (uint32_t c);
    
    const char *
    GetProcessPluginName ();
    
    void
    SetProcessPluginName (const char *plugin_name);
    
    uint32_t
    GetUserID();
    
    uint32_t
    GetGroupID();
    
    bool
    UserIDIsValid ();
    
    bool
    GroupIDIsValid ();
    
    void
    SetUserID (uint32_t uid);
    
    void
    SetGroupID (uint32_t gid);
    
    uint32_t
    GetEffectiveUserID();
    
    uint32_t
    GetEffectiveGroupID();
    
    bool
    EffectiveUserIDIsValid ();
    
    bool
    EffectiveGroupIDIsValid ();
    
    void
    SetEffectiveUserID (uint32_t uid);
    
    void
    SetEffectiveGroupID (uint32_t gid);
    
    lldb::pid_t
    GetParentProcessID ();
    
    void
    SetParentProcessID (lldb::pid_t pid);
    
    bool
    ParentProcessIDIsValid();
    
    
protected:
    friend class SBTarget;

    lldb_private::ProcessAttachInfo &
    ref ();
    
    ProcessAttachInfoSP m_opaque_sp;
};

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

    const lldb::SBTarget&
    operator = (const lldb::SBTarget& rhs);

    //------------------------------------------------------------------
    // Destructor
    //------------------------------------------------------------------
    ~SBTarget();

    bool
    IsValid() const;
    
    static const char *
    GetBroadcasterClassName ();

    lldb::SBProcess
    GetProcess ();

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
    /// @param[in] stop_at_endtry
    ///     If false do not stop the inferior at the entry point.
    ///
    /// @param[out]
    ///     An error object. Contains the reason if there is some failure.
    ///
    /// @return
    ///      A process object for the newly created process.
    //------------------------------------------------------------------
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
    SBProcess
    LaunchSimple (const char **argv, 
                  const char **envp,
                  const char *working_directory);
    
    SBProcess
    Launch (SBLaunchInfo &launch_info, SBError& error);
    
    SBProcess
    LoadCore (const char *core_file);

    SBProcess
    Attach (SBAttachInfo &attach_info, SBError& error);

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
    lldb::SBProcess
    AttachToProcessWithID (SBListener &listener,
                           lldb::pid_t pid,
                           lldb::SBError& error);

#if defined(__APPLE__)
    // We need to keep this around for a build or two since Xcode links
    // to the 32 bit version of this function. We will take it out soon.
    lldb::SBProcess
    AttachToProcessWithID (SBListener &listener,
                           ::pid_t pid,           // 32 bit int process ID
                           lldb::SBError& error); // DEPRECATED 
#endif
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
    lldb::SBProcess
    AttachToProcessWithName (SBListener &listener,
                             const char *name,
                             bool wait_for,
                             lldb::SBError& error);

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

    //------------------------------------------------------------------
    /// Set the base load address for a module section.
    ///
    /// @param[in] section
    ///     The section whose base load address will be set within this
    ///     target.
    ///
    /// @param[in] section_base_addr
    ///     The base address for the section.
    ///
    /// @return
    ///      An error to indicate success, fail, and any reason for 
    ///     failure.
    //------------------------------------------------------------------
    lldb::SBError
    SetSectionLoadAddress (lldb::SBSection section,
                           lldb::addr_t section_base_addr);
    
    //------------------------------------------------------------------
    /// Clear the base load address for a module section.
    ///
    /// @param[in] section
    ///     The section whose base load address will be cleared within
    ///     this target.
    ///
    /// @return
    ///      An error to indicate success, fail, and any reason for 
    ///     failure.
    //------------------------------------------------------------------
    lldb::SBError
    ClearSectionLoadAddress (lldb::SBSection section);
    
    //------------------------------------------------------------------
    /// Slide all file addresses for all module sections so that \a module
    /// appears to loaded at these slide addresses.
    /// 
    /// When you need all sections within a module to be loaded at a 
    /// rigid slide from the addresses found in the module object file,
    /// this function will allow you to easily and quickly slide all
    /// module sections.
    ///
    /// @param[in] module
    ///     The module to load.
    ///
    /// @param[in] sections_offset
    ///     An offset that will be applied to all section file addresses
    ///     (the virtual addresses found in the object file itself).
    ///
    /// @return
    ///     An error to indicate success, fail, and any reason for 
    ///     failure.
    //------------------------------------------------------------------
    lldb::SBError
    SetModuleLoadAddress (lldb::SBModule module,
                          int64_t sections_offset);
    

    //------------------------------------------------------------------
    /// The the section base load addresses for all sections in a module.
    /// 
    /// @param[in] module
    ///     The module to unload.
    ///
    /// @return
    ///     An error to indicate success, fail, and any reason for 
    ///     failure.
    //------------------------------------------------------------------
    lldb::SBError
    ClearModuleLoadAddress (lldb::SBModule module);

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
    lldb::SBSymbolContextList
    FindFunctions (const char *name, 
                   uint32_t name_type_mask = lldb::eFunctionNameTypeAny);

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
    lldb::SBValueList
    FindGlobalVariables (const char *name, 
                         uint32_t max_matches);

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
    lldb::SBValue
    FindFirstGlobalVariable (const char* name);
    
    void
    Clear ();

    lldb::SBAddress
    ResolveLoadAddress (lldb::addr_t vm_addr);

    SBSymbolContext
    ResolveSymbolContextForAddress (const SBAddress& addr, 
                                    uint32_t resolve_scope);

    lldb::SBBreakpoint
    BreakpointCreateByLocation (const char *file, uint32_t line);

    lldb::SBBreakpoint
    BreakpointCreateByLocation (const lldb::SBFileSpec &file_spec, uint32_t line);

    lldb::SBBreakpoint
    BreakpointCreateByName (const char *symbol_name, const char *module_name = NULL);

    // This version uses name_type_mask = eFunctionNameTypeAuto
    lldb::SBBreakpoint
    BreakpointCreateByName (const char *symbol_name, 
                            const SBFileSpecList &module_list, 
                            const SBFileSpecList &comp_unit_list);

    lldb::SBBreakpoint
    BreakpointCreateByName (const char *symbol_name,
                            uint32_t name_type_mask,           // Logical OR one or more FunctionNameType enum bits
                            const SBFileSpecList &module_list, 
                            const SBFileSpecList &comp_unit_list);

    lldb::SBBreakpoint
    BreakpointCreateByNames (const char *symbol_name[],
                             uint32_t num_names,
                             uint32_t name_type_mask,           // Logical OR one or more FunctionNameType enum bits
                             const SBFileSpecList &module_list,
                             const SBFileSpecList &comp_unit_list);

    lldb::SBBreakpoint
    BreakpointCreateByRegex (const char *symbol_name_regex, const char *module_name = NULL);
    
    lldb::SBBreakpoint
    BreakpointCreateByRegex (const char *symbol_name_regex, 
                             const SBFileSpecList &module_list, 
                             const SBFileSpecList &comp_unit_list);
    
    lldb::SBBreakpoint
    BreakpointCreateBySourceRegex (const char *source_regex, 
                                   const lldb::SBFileSpec &source_file, 
                                   const char *module_name = NULL);

    lldb::SBBreakpoint
    BreakpointCreateBySourceRegex (const char *source_regex, 
                                   const SBFileSpecList &module_list, 
                                   const lldb::SBFileSpecList &source_file);
    
    lldb::SBBreakpoint
    BreakpointCreateForException  (lldb::LanguageType language,
                                   bool catch_bp,
                                   bool throw_bp);

    lldb::SBBreakpoint
    BreakpointCreateByAddress (addr_t address);

    uint32_t
    GetNumBreakpoints () const;

    lldb::SBBreakpoint
    GetBreakpointAtIndex (uint32_t idx) const;

    bool
    BreakpointDelete (break_id_t break_id);

    lldb::SBBreakpoint
    FindBreakpointByID (break_id_t break_id);

    bool
    EnableAllBreakpoints ();

    bool
    DisableAllBreakpoints ();

    bool
    DeleteAllBreakpoints ();

    uint32_t
    GetNumWatchpoints () const;

    lldb::SBWatchpoint
    GetWatchpointAtIndex (uint32_t idx) const;

    bool
    DeleteWatchpoint (lldb::watch_id_t watch_id);

    lldb::SBWatchpoint
    FindWatchpointByID (lldb::watch_id_t watch_id);

    lldb::SBWatchpoint
    WatchAddress (lldb::addr_t addr, size_t size, bool read, bool write, SBError& error);

    bool
    EnableAllWatchpoints ();

    bool
    DisableAllWatchpoints ();

    bool
    DeleteAllWatchpoints ();

    lldb::SBBroadcaster
    GetBroadcaster () const;
    
    lldb::SBType
    FindFirstType (const char* type);
    
    lldb::SBTypeList
    FindTypes (const char* type);
    
    lldb::SBType
    GetBasicType(lldb::BasicType type);
    
    SBSourceManager
    GetSourceManager();
    
    lldb::SBInstructionList
    ReadInstructions (lldb::SBAddress base_addr, uint32_t count);

    lldb::SBInstructionList
    ReadInstructions (lldb::SBAddress base_addr, uint32_t count, const char *flavor_string);

    lldb::SBInstructionList
    GetInstructions (lldb::SBAddress base_addr, const void *buf, size_t size);
    
    // The "WithFlavor" is necessary to keep SWIG from getting confused about overloaded arguments when
    // using the buf + size -> Python Object magic.
    
    lldb::SBInstructionList
    GetInstructionsWithFlavor (lldb::SBAddress base_addr,  const char *flavor_string, const void *buf, size_t size);
    
    lldb::SBInstructionList
    GetInstructions (lldb::addr_t base_addr, const void *buf, size_t size);

    lldb::SBInstructionList
    GetInstructionsWithFlavor (lldb::addr_t base_addr, const char *flavor_string, const void *buf, size_t size);

    lldb::SBSymbolContextList
    FindSymbols (const char *name,
                 lldb::SymbolType type = eSymbolTypeAny);

    bool
    operator == (const lldb::SBTarget &rhs) const;

    bool
    operator != (const lldb::SBTarget &rhs) const;

    bool
    GetDescription (lldb::SBStream &description, lldb::DescriptionLevel description_level);

    lldb::SBValue
    EvaluateExpression (const char *expr, const SBExpressionOptions &options);

    lldb::addr_t
    GetStackRedZoneSize();
    
protected:
    friend class SBAddress;
    friend class SBBlock;
    friend class SBDebugger;
    friend class SBFunction;
    friend class SBInstruction;
    friend class SBModule;
    friend class SBProcess;
    friend class SBSection;
    friend class SBSourceManager;
    friend class SBSymbol;
    friend class SBValue;

    //------------------------------------------------------------------
    // Constructors are private, use static Target::Create function to
    // create an instance of this class.
    //------------------------------------------------------------------

    SBTarget (const lldb::TargetSP& target_sp);

    lldb::TargetSP
    GetSP () const;

    void
    SetSP (const lldb::TargetSP& target_sp);


private:
    //------------------------------------------------------------------
    // For Target only
    //------------------------------------------------------------------

    lldb::TargetSP m_opaque_sp;
};

} // namespace lldb

#endif  // LLDB_SBTarget_h_
