//===-- Target.h ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Target_h_
#define liblldb_Target_h_

// C Includes
// C++ Includes

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-public.h"
#include "lldb/Breakpoint/BreakpointList.h"
#include "lldb/Breakpoint/BreakpointLocationCollection.h"
#include "lldb/Breakpoint/WatchpointList.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/Broadcaster.h"
#include "lldb/Core/Event.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/UserSettingsController.h"
#include "lldb/Core/SourceManager.h"
#include "lldb/Expression/ClangPersistentVariables.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Interpreter/OptionValueBoolean.h"
#include "lldb/Interpreter/OptionValueEnumeration.h"
#include "lldb/Interpreter/OptionValueFileSpec.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/ABI.h"
#include "lldb/Target/ExecutionContextScope.h"
#include "lldb/Target/PathMappingList.h"
#include "lldb/Target/SectionLoadList.h"

namespace lldb_private {

extern OptionEnumValueElement g_dynamic_value_types[];

typedef enum InlineStrategy
{
    eInlineBreakpointsNever = 0,
    eInlineBreakpointsHeaders,
    eInlineBreakpointsAlways
} InlineStrategy;

//----------------------------------------------------------------------
// TargetProperties
//----------------------------------------------------------------------
class TargetProperties : public Properties
{
public:
    TargetProperties(Target *target);

    virtual
    ~TargetProperties();
    
    ArchSpec
    GetDefaultArchitecture () const;
    
    void
    SetDefaultArchitecture (const ArchSpec& arch);

    lldb::DynamicValueType
    GetPreferDynamicValue() const;
    
    bool
    GetDisableASLR () const;
    
    void
    SetDisableASLR (bool b);
    
    bool
    GetDisableSTDIO () const;
    
    void
    SetDisableSTDIO (bool b);
    
    InlineStrategy
    GetInlineStrategy () const;

    const char *
    GetArg0 () const;
    
    void
    SetArg0 (const char *arg);

    bool
    GetRunArguments (Args &args) const;
    
    void
    SetRunArguments (const Args &args);
    
    size_t
    GetEnvironmentAsArgs (Args &env) const;
    
    bool
    GetSkipPrologue() const;
    
    PathMappingList &
    GetSourcePathMap () const;
    
    FileSpecList &
    GetExecutableSearchPaths ();
    
    bool
    GetEnableSyntheticValue () const;
    
    uint32_t
    GetMaximumNumberOfChildrenToDisplay() const;
    
    uint32_t
    GetMaximumSizeOfStringSummary() const;
    
    FileSpec
    GetStandardInputPath () const;
    
    void
    SetStandardInputPath (const char *path);
    
    FileSpec
    GetStandardOutputPath () const;
    
    void
    SetStandardOutputPath (const char *path);
    
    FileSpec
    GetStandardErrorPath () const;
    
    void
    SetStandardErrorPath (const char *path);
    
    bool
    GetBreakpointsConsultPlatformAvoidList ();
    
    const char *
    GetExpressionPrefixContentsAsCString ();

};

typedef STD_SHARED_PTR(TargetProperties) TargetPropertiesSP;

class EvaluateExpressionOptions
{
public:
    static const uint32_t default_timeout = 500000;
    EvaluateExpressionOptions() :
        m_execution_policy(eExecutionPolicyOnlyWhenNeeded),
        m_coerce_to_id(false),
        m_unwind_on_error(true),
        m_keep_in_memory(false),
        m_run_others(true),
        m_use_dynamic(lldb::eNoDynamicValues),
        m_timeout_usec(default_timeout)
    {}
    
    ExecutionPolicy
    GetExecutionPolicy () const
    {
        return m_execution_policy;
    }
    
    EvaluateExpressionOptions&
    SetExecutionPolicy (ExecutionPolicy policy = eExecutionPolicyAlways)
    {
        m_execution_policy = policy;
        return *this;
    }
    
    bool
    DoesCoerceToId () const
    {
        return m_coerce_to_id;
    }
    
    EvaluateExpressionOptions&
    SetCoerceToId (bool coerce = true)
    {
        m_coerce_to_id = coerce;
        return *this;
    }
    
    bool
    DoesUnwindOnError () const
    {
        return m_unwind_on_error;
    }
    
    EvaluateExpressionOptions&
    SetUnwindOnError (bool unwind = false)
    {
        m_unwind_on_error = unwind;
        return *this;
    }
    
    bool
    DoesKeepInMemory () const
    {
        return m_keep_in_memory;
    }
    
    EvaluateExpressionOptions&
    SetKeepInMemory (bool keep = true)
    {
        m_keep_in_memory = keep;
        return *this;
    }
    
    lldb::DynamicValueType
    GetUseDynamic () const
    {
        return m_use_dynamic;
    }
    
    EvaluateExpressionOptions&
    SetUseDynamic (lldb::DynamicValueType dynamic = lldb::eDynamicCanRunTarget)
    {
        m_use_dynamic = dynamic;
        return *this;
    }
    
    uint32_t
    GetTimeoutUsec () const
    {
        return m_timeout_usec;
    }
    
    EvaluateExpressionOptions&
    SetTimeoutUsec (uint32_t timeout = 0)
    {
        m_timeout_usec = timeout;
        return *this;
    }
    
    bool
    GetRunOthers () const
    {
        return m_run_others;
    }
    
    EvaluateExpressionOptions&
    SetRunOthers (bool run_others = true)
    {
        m_run_others = run_others;
        return *this;
    }
    
private:
    ExecutionPolicy m_execution_policy;
    bool m_coerce_to_id;
    bool m_unwind_on_error;
    bool m_keep_in_memory;
    bool m_run_others;
    lldb::DynamicValueType m_use_dynamic;
    uint32_t m_timeout_usec;
};

//----------------------------------------------------------------------
// Target
//----------------------------------------------------------------------
class Target :
    public STD_ENABLE_SHARED_FROM_THIS(Target),
    public TargetProperties,
    public Broadcaster,
    public ExecutionContextScope,
    public ModuleList::Notifier
{
public:
    friend class TargetList;

    //------------------------------------------------------------------
    /// Broadcaster event bits definitions.
    //------------------------------------------------------------------
    enum
    {
        eBroadcastBitBreakpointChanged  = (1 << 0),
        eBroadcastBitModulesLoaded      = (1 << 1),
        eBroadcastBitModulesUnloaded    = (1 << 2)
    };
    
    // These two functions fill out the Broadcaster interface:
    
    static ConstString &GetStaticBroadcasterClass ();

    virtual ConstString &GetBroadcasterClass() const
    {
        return GetStaticBroadcasterClass();
    }

    // This event data class is for use by the TargetList to broadcast new target notifications.
    class TargetEventData : public EventData
    {
    public:

        static const ConstString &
        GetFlavorString ();

        virtual const ConstString &
        GetFlavor () const;

        TargetEventData (const lldb::TargetSP &new_target_sp);
        
        lldb::TargetSP &
        GetTarget()
        {
            return m_target_sp;
        }

        virtual
        ~TargetEventData();
        
        virtual void
        Dump (Stream *s) const;

        static const lldb::TargetSP
        GetTargetFromEvent (const lldb::EventSP &event_sp);
        
        static const TargetEventData *
        GetEventDataFromEvent (const Event *event_sp);

    private:
        lldb::TargetSP m_target_sp;

        DISALLOW_COPY_AND_ASSIGN (TargetEventData);
    };
    
    static void
    SettingsInitialize ();

    static void
    SettingsTerminate ();

//    static lldb::UserSettingsControllerSP &
//    GetSettingsController ();

    static FileSpecList
    GetDefaultExecutableSearchPaths ();

    static ArchSpec
    GetDefaultArchitecture ();

    static void
    SetDefaultArchitecture (const ArchSpec &arch);

//    void
//    UpdateInstanceName ();

    lldb::ModuleSP
    GetSharedModule (const ModuleSpec &module_spec,
                     Error *error_ptr = NULL);

    //----------------------------------------------------------------------
    // Settings accessors
    //----------------------------------------------------------------------

    static const TargetPropertiesSP &
    GetGlobalProperties();


private:
    //------------------------------------------------------------------
    /// Construct with optional file and arch.
    ///
    /// This member is private. Clients must use
    /// TargetList::CreateTarget(const FileSpec*, const ArchSpec*)
    /// so all targets can be tracked from the central target list.
    ///
    /// @see TargetList::CreateTarget(const FileSpec*, const ArchSpec*)
    //------------------------------------------------------------------
    Target (Debugger &debugger,
            const ArchSpec &target_arch,
            const lldb::PlatformSP &platform_sp);

    // Helper function.
    bool
    ProcessIsValid ();

public:
    ~Target();

    Mutex &
    GetAPIMutex ()
    {
        return m_mutex;
    }

    void
    DeleteCurrentProcess ();

    //------------------------------------------------------------------
    /// Dump a description of this object to a Stream.
    ///
    /// Dump a description of the contents of this object to the
    /// supplied stream \a s. The dumped content will be only what has
    /// been loaded or parsed up to this point at which this function
    /// is called, so this is a good way to see what has been parsed
    /// in a target.
    ///
    /// @param[in] s
    ///     The stream to which to dump the object descripton.
    //------------------------------------------------------------------
    void
    Dump (Stream *s, lldb::DescriptionLevel description_level);

    const lldb::ProcessSP &
    CreateProcess (Listener &listener, 
                   const char *plugin_name,
                   const FileSpec *crash_file);

    const lldb::ProcessSP &
    GetProcessSP () const;

    bool
    IsValid()
    {
        return m_valid;
    }

    void
    Destroy();

    //------------------------------------------------------------------
    // This part handles the breakpoints.
    //------------------------------------------------------------------

    BreakpointList &
    GetBreakpointList(bool internal = false);

    const BreakpointList &
    GetBreakpointList(bool internal = false) const;
    
    lldb::BreakpointSP
    GetLastCreatedBreakpoint ()
    {
        return m_last_created_breakpoint;
    }

    lldb::BreakpointSP
    GetBreakpointByID (lldb::break_id_t break_id);

    // Use this to create a file and line breakpoint to a given module or all module it is NULL
    lldb::BreakpointSP
    CreateBreakpoint (const FileSpecList *containingModules,
                      const FileSpec &file,
                      uint32_t line_no,
                      LazyBool check_inlines = eLazyBoolCalculate,
                      LazyBool skip_prologue = eLazyBoolCalculate,
                      bool internal = false);

    // Use this to create breakpoint that matches regex against the source lines in files given in source_file_list:
    lldb::BreakpointSP
    CreateSourceRegexBreakpoint (const FileSpecList *containingModules,
                                 const FileSpecList *source_file_list,
                                 RegularExpression &source_regex,
                                 bool internal = false);

    // Use this to create a breakpoint from a load address
    lldb::BreakpointSP
    CreateBreakpoint (lldb::addr_t load_addr,
                      bool internal = false);

    // Use this to create Address breakpoints:
    lldb::BreakpointSP
    CreateBreakpoint (Address &addr,
                      bool internal = false);

    // Use this to create a function breakpoint by regexp in containingModule/containingSourceFiles, or all modules if it is NULL
    // When "skip_prologue is set to eLazyBoolCalculate, we use the current target 
    // setting, else we use the values passed in
    lldb::BreakpointSP
    CreateFuncRegexBreakpoint (const FileSpecList *containingModules,
                               const FileSpecList *containingSourceFiles,
                               RegularExpression &func_regexp,
                               LazyBool skip_prologue = eLazyBoolCalculate,
                               bool internal = false);

    // Use this to create a function breakpoint by name in containingModule, or all modules if it is NULL
    // When "skip_prologue is set to eLazyBoolCalculate, we use the current target 
    // setting, else we use the values passed in
    lldb::BreakpointSP
    CreateBreakpoint (const FileSpecList *containingModules,
                      const FileSpecList *containingSourceFiles,
                      const char *func_name,
                      uint32_t func_name_type_mask, 
                      LazyBool skip_prologue = eLazyBoolCalculate,
                      bool internal = false);
                      
    lldb::BreakpointSP
    CreateExceptionBreakpoint (enum lldb::LanguageType language, bool catch_bp, bool throw_bp, bool internal = false);
    
    // This is the same as the func_name breakpoint except that you can specify a vector of names.  This is cheaper
    // than a regular expression breakpoint in the case where you just want to set a breakpoint on a set of names
    // you already know.
    lldb::BreakpointSP
    CreateBreakpoint (const FileSpecList *containingModules,
                      const FileSpecList *containingSourceFiles,
                      const char *func_names[],
                      size_t num_names, 
                      uint32_t func_name_type_mask, 
                      LazyBool skip_prologue = eLazyBoolCalculate,
                      bool internal = false);

    lldb::BreakpointSP
    CreateBreakpoint (const FileSpecList *containingModules,
                      const FileSpecList *containingSourceFiles,
                      const std::vector<std::string> &func_names,
                      uint32_t func_name_type_mask,
                      LazyBool skip_prologue = eLazyBoolCalculate,
                      bool internal = false);


    // Use this to create a general breakpoint:
    lldb::BreakpointSP
    CreateBreakpoint (lldb::SearchFilterSP &filter_sp,
                      lldb::BreakpointResolverSP &resolver_sp,
                      bool internal = false);

    // Use this to create a watchpoint:
    lldb::WatchpointSP
    CreateWatchpoint (lldb::addr_t addr,
                      size_t size,
                      const ClangASTType *type,
                      uint32_t kind,
                      Error &error);

    lldb::WatchpointSP
    GetLastCreatedWatchpoint ()
    {
        return m_last_created_watchpoint;
    }

    WatchpointList &
    GetWatchpointList()
    {
        return m_watchpoint_list;
    }

    void
    RemoveAllBreakpoints (bool internal_also = false);

    void
    DisableAllBreakpoints (bool internal_also = false);

    void
    EnableAllBreakpoints (bool internal_also = false);

    bool
    DisableBreakpointByID (lldb::break_id_t break_id);

    bool
    EnableBreakpointByID (lldb::break_id_t break_id);

    bool
    RemoveBreakpointByID (lldb::break_id_t break_id);

    // The flag 'end_to_end', default to true, signifies that the operation is
    // performed end to end, for both the debugger and the debuggee.

    bool
    RemoveAllWatchpoints (bool end_to_end = true);

    bool
    DisableAllWatchpoints (bool end_to_end = true);

    bool
    EnableAllWatchpoints (bool end_to_end = true);

    bool
    ClearAllWatchpointHitCounts ();

    bool
    IgnoreAllWatchpoints (uint32_t ignore_count);

    bool
    DisableWatchpointByID (lldb::watch_id_t watch_id);

    bool
    EnableWatchpointByID (lldb::watch_id_t watch_id);

    bool
    RemoveWatchpointByID (lldb::watch_id_t watch_id);

    bool
    IgnoreWatchpointByID (lldb::watch_id_t watch_id, uint32_t ignore_count);

    //------------------------------------------------------------------
    /// Get \a load_addr as a callable code load address for this target
    ///
    /// Take \a load_addr and potentially add any address bits that are 
    /// needed to make the address callable. For ARM this can set bit
    /// zero (if it already isn't) if \a load_addr is a thumb function.
    /// If \a addr_class is set to eAddressClassInvalid, then the address
    /// adjustment will always happen. If it is set to an address class
    /// that doesn't have code in it, LLDB_INVALID_ADDRESS will be 
    /// returned.
    //------------------------------------------------------------------
    lldb::addr_t
    GetCallableLoadAddress (lldb::addr_t load_addr, lldb::AddressClass addr_class = lldb::eAddressClassInvalid) const;

    //------------------------------------------------------------------
    /// Get \a load_addr as an opcode for this target.
    ///
    /// Take \a load_addr and potentially strip any address bits that are 
    /// needed to make the address point to an opcode. For ARM this can 
    /// clear bit zero (if it already isn't) if \a load_addr is a 
    /// thumb function and load_addr is in code.
    /// If \a addr_class is set to eAddressClassInvalid, then the address
    /// adjustment will always happen. If it is set to an address class
    /// that doesn't have code in it, LLDB_INVALID_ADDRESS will be 
    /// returned.
    //------------------------------------------------------------------
    lldb::addr_t
    GetOpcodeLoadAddress (lldb::addr_t load_addr, lldb::AddressClass addr_class = lldb::eAddressClassInvalid) const;

protected:
    //------------------------------------------------------------------
    /// Implementing of ModuleList::Notifier.
    //------------------------------------------------------------------
    
    virtual void
    ModuleAdded (const ModuleList& module_list, const lldb::ModuleSP& module_sp);
    
    virtual void
    ModuleRemoved (const ModuleList& module_list, const lldb::ModuleSP& module_sp);
    
    virtual void
    ModuleUpdated (const ModuleList& module_list,
                   const lldb::ModuleSP& old_module_sp,
                   const lldb::ModuleSP& new_module_sp);
    virtual void
    WillClearList (const ModuleList& module_list);

public:
    
    void
    ModulesDidLoad (ModuleList &module_list);

    void
    ModulesDidUnload (ModuleList &module_list);
    
    //------------------------------------------------------------------
    /// Gets the module for the main executable.
    ///
    /// Each process has a notion of a main executable that is the file
    /// that will be executed or attached to. Executable files can have
    /// dependent modules that are discovered from the object files, or
    /// discovered at runtime as things are dynamically loaded.
    ///
    /// @return
    ///     The shared pointer to the executable module which can
    ///     contains a NULL Module object if no executable has been
    ///     set.
    ///
    /// @see DynamicLoader
    /// @see ObjectFile::GetDependentModules (FileSpecList&)
    /// @see Process::SetExecutableModule(lldb::ModuleSP&)
    //------------------------------------------------------------------
    lldb::ModuleSP
    GetExecutableModule ();

    Module*
    GetExecutableModulePointer ();

    //------------------------------------------------------------------
    /// Set the main executable module.
    ///
    /// Each process has a notion of a main executable that is the file
    /// that will be executed or attached to. Executable files can have
    /// dependent modules that are discovered from the object files, or
    /// discovered at runtime as things are dynamically loaded.
    ///
    /// Setting the executable causes any of the current dependant
    /// image information to be cleared and replaced with the static
    /// dependent image information found by calling
    /// ObjectFile::GetDependentModules (FileSpecList&) on the main
    /// executable and any modules on which it depends. Calling
    /// Process::GetImages() will return the newly found images that
    /// were obtained from all of the object files.
    ///
    /// @param[in] module_sp
    ///     A shared pointer reference to the module that will become
    ///     the main executable for this process.
    ///
    /// @param[in] get_dependent_files
    ///     If \b true then ask the object files to track down any
    ///     known dependent files.
    ///
    /// @see ObjectFile::GetDependentModules (FileSpecList&)
    /// @see Process::GetImages()
    //------------------------------------------------------------------
    void
    SetExecutableModule (lldb::ModuleSP& module_sp, bool get_dependent_files);

    //------------------------------------------------------------------
    /// Get accessor for the images for this process.
    ///
    /// Each process has a notion of a main executable that is the file
    /// that will be executed or attached to. Executable files can have
    /// dependent modules that are discovered from the object files, or
    /// discovered at runtime as things are dynamically loaded. After
    /// a main executable has been set, the images will contain a list
    /// of all the files that the executable depends upon as far as the
    /// object files know. These images will usually contain valid file
    /// virtual addresses only. When the process is launched or attached
    /// to, the DynamicLoader plug-in will discover where these images
    /// were loaded in memory and will resolve the load virtual
    /// addresses is each image, and also in images that are loaded by
    /// code.
    ///
    /// @return
    ///     A list of Module objects in a module list.
    //------------------------------------------------------------------
    const ModuleList&
    GetImages () const
    {
        return m_images;
    }
    
    ModuleList&
    GetImages ()
    {
        return m_images;
    }
    
    //------------------------------------------------------------------
    /// Return whether this FileSpec corresponds to a module that should be considered for general searches.
    ///
    /// This API will be consulted by the SearchFilterForNonModuleSpecificSearches
    /// and any module that returns \b true will not be searched.  Note the
    /// SearchFilterForNonModuleSpecificSearches is the search filter that
    /// gets used in the CreateBreakpoint calls when no modules is provided.
    ///
    /// The target call at present just consults the Platform's call of the
    /// same name.
    /// 
    /// @param[in] module_sp
    ///     A shared pointer reference to the module that checked.
    ///
    /// @return \b true if the module should be excluded, \b false otherwise.
    //------------------------------------------------------------------
    bool
    ModuleIsExcludedForNonModuleSpecificSearches (const FileSpec &module_spec);
    
    //------------------------------------------------------------------
    /// Return whether this module should be considered for general searches.
    ///
    /// This API will be consulted by the SearchFilterForNonModuleSpecificSearches
    /// and any module that returns \b true will not be searched.  Note the
    /// SearchFilterForNonModuleSpecificSearches is the search filter that
    /// gets used in the CreateBreakpoint calls when no modules is provided.
    ///
    /// The target call at present just consults the Platform's call of the
    /// same name.
    ///
    /// FIXME: When we get time we should add a way for the user to set modules that they
    /// don't want searched, in addition to or instead of the platform ones.
    /// 
    /// @param[in] module_sp
    ///     A shared pointer reference to the module that checked.
    ///
    /// @return \b true if the module should be excluded, \b false otherwise.
    //------------------------------------------------------------------
    bool
    ModuleIsExcludedForNonModuleSpecificSearches (const lldb::ModuleSP &module_sp);

    ArchSpec &
    GetArchitecture ()
    {
        return m_arch;
    }
    
    const ArchSpec &
    GetArchitecture () const
    {
        return m_arch;
    }
    
    //------------------------------------------------------------------
    /// Set the architecture for this target.
    ///
    /// If the current target has no Images read in, then this just sets the architecture, which will
    /// be used to select the architecture of the ExecutableModule when that is set.
    /// If the current target has an ExecutableModule, then calling SetArchitecture with a different
    /// architecture from the currently selected one will reset the ExecutableModule to that slice
    /// of the file backing the ExecutableModule.  If the file backing the ExecutableModule does not
    /// contain a fork of this architecture, then this code will return false, and the architecture
    /// won't be changed.
    /// If the input arch_spec is the same as the already set architecture, this is a no-op.
    ///
    /// @param[in] arch_spec
    ///     The new architecture.
    ///
    /// @return
    ///     \b true if the architecture was successfully set, \bfalse otherwise.
    //------------------------------------------------------------------
    bool
    SetArchitecture (const ArchSpec &arch_spec);

    Debugger &
    GetDebugger ()
    {
        return m_debugger;
    }

    size_t
    ReadMemoryFromFileCache (const Address& addr, 
                             void *dst, 
                             size_t dst_len, 
                             Error &error);

    // Reading memory through the target allows us to skip going to the process
    // for reading memory if possible and it allows us to try and read from 
    // any constant sections in our object files on disk. If you always want
    // live program memory, read straight from the process. If you possibly 
    // want to read from const sections in object files, read from the target.
    // This version of ReadMemory will try and read memory from the process
    // if the process is alive. The order is:
    // 1 - if (prefer_file_cache == true) then read from object file cache
    // 2 - if there is a valid process, try and read from its memory
    // 3 - if (prefer_file_cache == false) then read from object file cache
    size_t
    ReadMemory (const Address& addr,
                bool prefer_file_cache,
                void *dst,
                size_t dst_len,
                Error &error,
                lldb::addr_t *load_addr_ptr = NULL);

    size_t
    ReadScalarIntegerFromMemory (const Address& addr, 
                                 bool prefer_file_cache,
                                 uint32_t byte_size, 
                                 bool is_signed, 
                                 Scalar &scalar, 
                                 Error &error);

    uint64_t
    ReadUnsignedIntegerFromMemory (const Address& addr, 
                                   bool prefer_file_cache,
                                   size_t integer_byte_size, 
                                   uint64_t fail_value, 
                                   Error &error);

    bool
    ReadPointerFromMemory (const Address& addr, 
                           bool prefer_file_cache,
                           Error &error,
                           Address &pointer_addr);

    SectionLoadList&
    GetSectionLoadList()
    {
        return m_section_load_list;
    }

    const SectionLoadList&
    GetSectionLoadList() const
    {
        return m_section_load_list;
    }

    static Target *
    GetTargetFromContexts (const ExecutionContext *exe_ctx_ptr, 
                           const SymbolContext *sc_ptr);

    //------------------------------------------------------------------
    // lldb::ExecutionContextScope pure virtual functions
    //------------------------------------------------------------------
    virtual lldb::TargetSP
    CalculateTarget ();
    
    virtual lldb::ProcessSP
    CalculateProcess ();
    
    virtual lldb::ThreadSP
    CalculateThread ();
    
    virtual lldb::StackFrameSP
    CalculateStackFrame ();

    virtual void
    CalculateExecutionContext (ExecutionContext &exe_ctx);

    PathMappingList &
    GetImageSearchPathList ();
    
    ClangASTContext *
    GetScratchClangASTContext(bool create_on_demand=true);
    
    ClangASTImporter *
    GetClangASTImporter();
    
    
    // Since expressions results can persist beyond the lifetime of a process,
    // and the const expression results are available after a process is gone,
    // we provide a way for expressions to be evaluated from the Target itself.
    // If an expression is going to be run, then it should have a frame filled
    // in in th execution context. 
    ExecutionResults
    EvaluateExpression (const char *expression,
                        StackFrame *frame,
                        lldb::ValueObjectSP &result_valobj_sp,
                        const EvaluateExpressionOptions& options = EvaluateExpressionOptions());

    ClangPersistentVariables &
    GetPersistentVariables()
    {
        return m_persistent_variables;
    }

    //------------------------------------------------------------------
    // Target Stop Hooks
    //------------------------------------------------------------------
    class StopHook : public UserID
    {
    public:
        ~StopHook ();
        
        StopHook (const StopHook &rhs);
                
        StringList *
        GetCommandPointer ()
        {
            return &m_commands;
        }
        
        const StringList &
        GetCommands()
        {
            return m_commands;
        }
        
        lldb::TargetSP &
        GetTarget()
        {
            return m_target_sp;
        }
        
        void
        SetCommands (StringList &in_commands)
        {
            m_commands = in_commands;
        }
        
        // Set the specifier.  The stop hook will own the specifier, and is responsible for deleting it when we're done.
        void
        SetSpecifier (SymbolContextSpecifier *specifier)
        {
            m_specifier_sp.reset (specifier);
        }
        
        SymbolContextSpecifier *
        GetSpecifier ()
        {
            return m_specifier_sp.get();
        }
        
        // Set the Thread Specifier.  The stop hook will own the thread specifier, and is responsible for deleting it when we're done.
        void
        SetThreadSpecifier (ThreadSpec *specifier);
        
        ThreadSpec *
        GetThreadSpecifier()
        {
            return m_thread_spec_ap.get();
        }
        
        bool
        IsActive()
        {
            return m_active;
        }
        
        void
        SetIsActive (bool is_active)
        {
            m_active = is_active;
        }
        
        void
        GetDescription (Stream *s, lldb::DescriptionLevel level) const;
        
    private:
        lldb::TargetSP m_target_sp;
        StringList   m_commands;
        lldb::SymbolContextSpecifierSP m_specifier_sp;
        std::auto_ptr<ThreadSpec> m_thread_spec_ap;
        bool m_active;
        
        // Use AddStopHook to make a new empty stop hook.  The GetCommandPointer and fill it with commands,
        // and SetSpecifier to set the specifier shared pointer (can be null, that will match anything.)
        StopHook (lldb::TargetSP target_sp, lldb::user_id_t uid);
        friend class Target;
    };
    typedef STD_SHARED_PTR(StopHook) StopHookSP;
    
    // Add an empty stop hook to the Target's stop hook list, and returns a shared pointer to it in new_hook.  
    // Returns the id of the new hook.        
    lldb::user_id_t
    AddStopHook (StopHookSP &new_hook);
    
    void
    RunStopHooks ();
    
    size_t
    GetStopHookSize();
    
    bool
    SetSuppresStopHooks (bool suppress)
    {
        bool old_value = m_suppress_stop_hooks;
        m_suppress_stop_hooks = suppress;
        return old_value;
    }
    
    bool
    GetSuppressStopHooks ()
    {
        return m_suppress_stop_hooks;
    }
    
    bool
    SetSuppressSyntheticValue (bool suppress)
    {
        bool old_value = m_suppress_synthetic_value;
        m_suppress_synthetic_value = suppress;
        return old_value;
    }
    
    bool
    GetSuppressSyntheticValue ()
    {
        return m_suppress_synthetic_value;
    }
    
//    StopHookSP &
//    GetStopHookByIndex (size_t index);
//    
    bool
    RemoveStopHookByID (lldb::user_id_t uid);
    
    void
    RemoveAllStopHooks ();
    
    StopHookSP
    GetStopHookByID (lldb::user_id_t uid);
    
    bool
    SetStopHookActiveStateByID (lldb::user_id_t uid, bool active_state);
    
    void
    SetAllStopHooksActiveState (bool active_state);
    
    size_t GetNumStopHooks () const
    {
        return m_stop_hooks.size();
    }
    
    StopHookSP
    GetStopHookAtIndex (size_t index)
    {
        if (index >= GetNumStopHooks())
            return StopHookSP();
        StopHookCollection::iterator pos = m_stop_hooks.begin();
        
        while (index > 0)
        {
            pos++;
            index--;
        }
        return (*pos).second;
    }
    
    lldb::PlatformSP
    GetPlatform ()
    {
        return m_platform_sp;
    }

    void
    SetPlatform (const lldb::PlatformSP &platform_sp)
    {
        m_platform_sp = platform_sp;
    }

    SourceManager &
    GetSourceManager ()
    {
        return m_source_manager;
    }

    //------------------------------------------------------------------
    // Methods.
    //------------------------------------------------------------------
    lldb::SearchFilterSP
    GetSearchFilterForModule (const FileSpec *containingModule);

    lldb::SearchFilterSP
    GetSearchFilterForModuleList (const FileSpecList *containingModuleList);
    
    lldb::SearchFilterSP
    GetSearchFilterForModuleAndCUList (const FileSpecList *containingModules, const FileSpecList *containingSourceFiles);

protected:
    //------------------------------------------------------------------
    // Member variables.
    //------------------------------------------------------------------
    Debugger &      m_debugger;
    lldb::PlatformSP m_platform_sp;     ///< The platform for this target.
    Mutex           m_mutex;            ///< An API mutex that is used by the lldb::SB* classes make the SB interface thread safe
    ArchSpec        m_arch;
    ModuleList      m_images;           ///< The list of images for this process (shared libraries and anything dynamically loaded).
    SectionLoadList m_section_load_list;
    BreakpointList  m_breakpoint_list;
    BreakpointList  m_internal_breakpoint_list;
    lldb::BreakpointSP m_last_created_breakpoint;
    WatchpointList  m_watchpoint_list;
    lldb::WatchpointSP m_last_created_watchpoint;
    // We want to tightly control the process destruction process so
    // we can correctly tear down everything that we need to, so the only
    // class that knows about the process lifespan is this target class.
    lldb::ProcessSP m_process_sp;
    bool m_valid;
    lldb::SearchFilterSP  m_search_filter_sp;
    PathMappingList m_image_search_paths;
    std::auto_ptr<ClangASTContext> m_scratch_ast_context_ap;
    std::auto_ptr<ClangASTSource> m_scratch_ast_source_ap;
    std::auto_ptr<ClangASTImporter> m_ast_importer_ap;
    ClangPersistentVariables m_persistent_variables;      ///< These are the persistent variables associated with this process for the expression parser.

    SourceManager m_source_manager;

    typedef std::map<lldb::user_id_t, StopHookSP> StopHookCollection;
    StopHookCollection      m_stop_hooks;
    lldb::user_id_t         m_stop_hook_next_id;
    bool                    m_suppress_stop_hooks;
    bool                    m_suppress_synthetic_value;
    
    static void
    ImageSearchPathsChanged (const PathMappingList &path_list,
                             void *baton);

private:
    DISALLOW_COPY_AND_ASSIGN (Target);
};

} // namespace lldb_private

#endif  // liblldb_Target_h_
