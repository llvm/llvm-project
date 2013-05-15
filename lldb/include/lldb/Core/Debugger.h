//===-- Debugger.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Debugger_h_
#define liblldb_Debugger_h_
#if defined(__cplusplus)


#include <stdint.h>
#include <unistd.h>

#include <stack>

#include "lldb/lldb-public.h"

#include "lldb/API/SBDefines.h"

#include "lldb/Core/Broadcaster.h"
#include "lldb/Core/Communication.h"
#include "lldb/Core/InputReaderStack.h"
#include "lldb/Core/Listener.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/SourceManager.h"
#include "lldb/Core/UserID.h"
#include "lldb/Core/UserSettingsController.h"
#include "lldb/DataFormatters/FormatManager.h"
#include "lldb/Host/Terminal.h"
#include "lldb/Interpreter/OptionValueProperties.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/TargetList.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class Debugger Debugger.h "lldb/Core/Debugger.h"
/// @brief A class to manage flag bits.
///
/// Provides a global root objects for the debugger core.
//----------------------------------------------------------------------


class Debugger :
    public std::enable_shared_from_this<Debugger>,
    public UserID,
    public Properties,
    public BroadcasterManager
{
friend class SourceManager;  // For GetSourceFileCache.

public:

    static lldb::DebuggerSP
    CreateInstance (lldb::LogOutputCallback log_callback = NULL, void *baton = NULL);

    static lldb::TargetSP
    FindTargetWithProcessID (lldb::pid_t pid);
    
    static lldb::TargetSP
    FindTargetWithProcess (Process *process);

    static void
    Initialize ();
    
    static void 
    Terminate ();
    
    static void
    SettingsInitialize ();
    
    static void
    SettingsTerminate ();
    
    static void
    Destroy (lldb::DebuggerSP &debugger_sp);

    virtual
    ~Debugger ();
    
    void Clear();

    bool
    GetAsyncExecution ();

    void
    SetAsyncExecution (bool async);

    File &
    GetInputFile ()
    {
        return m_input_file.GetFile();
    }

    File &
    GetOutputFile ()
    {
        return m_output_file.GetFile();
    }

    File &
    GetErrorFile ()
    {
        return m_error_file.GetFile();
    }
    
    void
    SetInputFileHandle (FILE *fh, bool tranfer_ownership);

    void
    SetOutputFileHandle (FILE *fh, bool tranfer_ownership);

    void
    SetErrorFileHandle (FILE *fh, bool tranfer_ownership);
    
    void
    SaveInputTerminalState();
    
    void
    RestoreInputTerminalState();

    Stream&
    GetOutputStream ()
    {
        return m_output_file;
    }

    Stream&
    GetErrorStream ()
    {
        return m_error_file;
    }

    lldb::StreamSP
    GetAsyncOutputStream ();
    
    lldb::StreamSP
    GetAsyncErrorStream ();
    
    CommandInterpreter &
    GetCommandInterpreter ()
    {
        assert (m_command_interpreter_ap.get());
        return *m_command_interpreter_ap;
    }

    Listener &
    GetListener ()
    {
        return m_listener;
    }

    // This returns the Debugger's scratch source manager.  It won't be able to look up files in debug
    // information, but it can look up files by absolute path and display them to you.
    // To get the target's source manager, call GetSourceManager on the target instead.
    SourceManager &
    GetSourceManager ();

public:
    
    lldb::TargetSP
    GetSelectedTarget ()
    {
        return m_target_list.GetSelectedTarget ();
    }

    ExecutionContext
    GetSelectedExecutionContext();
    //------------------------------------------------------------------
    /// Get accessor for the target list.
    ///
    /// The target list is part of the global debugger object. This
    /// the single debugger shared instance to control where targets
    /// get created and to allow for tracking and searching for targets
    /// based on certain criteria.
    ///
    /// @return
    ///     A global shared target list.
    //------------------------------------------------------------------
    TargetList &
    GetTargetList ()
    {
        return m_target_list;
    }

    PlatformList &
    GetPlatformList ()
    {
        return m_platform_list;
    }

    void
    DispatchInputInterrupt ();

    void
    DispatchInputEndOfFile ();

    void
    DispatchInput (const char *bytes, size_t bytes_len);

    void
    WriteToDefaultReader (const char *bytes, size_t bytes_len);

    void
    PushInputReader (const lldb::InputReaderSP& reader_sp);

    bool
    PopInputReader (const lldb::InputReaderSP& reader_sp);

    void
    NotifyTopInputReader (lldb::InputReaderAction notification);

    bool
    InputReaderIsTopReader (const lldb::InputReaderSP& reader_sp);
    
    static lldb::DebuggerSP
    FindDebuggerWithID (lldb::user_id_t id);
    
    static lldb::DebuggerSP
    FindDebuggerWithInstanceName (const ConstString &instance_name);
    
    static size_t
    GetNumDebuggers();
    
    static lldb::DebuggerSP
    GetDebuggerAtIndex (size_t index);

    static bool
    FormatPrompt (const char *format,
                  const SymbolContext *sc,
                  const ExecutionContext *exe_ctx,
                  const Address *addr,
                  Stream &s,
                  const char **end,
                  ValueObject* valobj = NULL);


    void
    CleanUpInputReaders ();

    static int
    TestDebuggerRefCount ();

    bool
    GetCloseInputOnEOF () const;
    
    void
    SetCloseInputOnEOF (bool b);
    
    bool
    EnableLog (const char *channel, const char **categories, const char *log_file, uint32_t log_options, Stream &error_stream);

    void
    SetLoggingCallback (lldb::LogOutputCallback log_callback, void *baton);
    

    //----------------------------------------------------------------------
    // Properties Functions
    //----------------------------------------------------------------------
    enum StopDisassemblyType
    {
        eStopDisassemblyTypeNever = 0,
        eStopDisassemblyTypeNoSource,
        eStopDisassemblyTypeAlways
    };
    
    virtual Error
    SetPropertyValue (const ExecutionContext *exe_ctx,
                      VarSetOperationType op,
                      const char *property_path,
                      const char *value);

    bool
    GetAutoConfirm () const;
    
    const char *
    GetFrameFormat() const;
    
    const char *
    GetThreadFormat() const;
    
    lldb::ScriptLanguage
    GetScriptLanguage() const;
    
    bool
    SetScriptLanguage (lldb::ScriptLanguage script_lang);
    
    uint32_t
    GetTerminalWidth () const;
    
    bool
    SetTerminalWidth (uint32_t term_width);
    
    const char *
    GetPrompt() const;
    
    void
    SetPrompt(const char *p);
    
    bool
    GetUseExternalEditor () const;
    
    bool
    SetUseExternalEditor (bool use_external_editor_p);
    
    uint32_t
    GetStopSourceLineCount (bool before) const;
    
    StopDisassemblyType
    GetStopDisassemblyDisplay () const;
    
    uint32_t
    GetDisassemblyLineCount () const;
    
    bool
    GetNotifyVoid () const;

    
    const ConstString &
    GetInstanceName()
    {
        return m_instance_name;
    }
    
    typedef bool (*LLDBCommandPluginInit) (lldb::SBDebugger& debugger);
    
    bool
    LoadPlugin (const FileSpec& spec, Error& error);

protected:

    static void
    DispatchInputCallback (void *baton, const void *bytes, size_t bytes_len);

    lldb::InputReaderSP
    GetCurrentInputReader ();
    
    void
    ActivateInputReader (const lldb::InputReaderSP &reader_sp);

    bool
    CheckIfTopInputReaderIsDone ();
    
    SourceManager::SourceFileCache &
    GetSourceFileCache ()
    {
        return m_source_file_cache;
    }
    Communication m_input_comm;
    StreamFile m_input_file;
    StreamFile m_output_file;
    StreamFile m_error_file;
    TerminalState m_terminal_state;
    TargetList m_target_list;
    PlatformList m_platform_list;
    Listener m_listener;
    std::unique_ptr<SourceManager> m_source_manager_ap;    // This is a scratch source manager that we return if we have no targets.
    SourceManager::SourceFileCache m_source_file_cache; // All the source managers for targets created in this debugger used this shared
                                                        // source file cache.
    std::unique_ptr<CommandInterpreter> m_command_interpreter_ap;

    InputReaderStack m_input_reader_stack;
    std::string m_input_reader_data;
    typedef std::map<std::string, lldb::StreamWP> LogStreamMap;
    LogStreamMap m_log_streams;
    lldb::StreamSP m_log_callback_stream_sp;
    ConstString m_instance_name;
    typedef std::vector<lldb::DynamicLibrarySP> LoadedPluginsList;
    LoadedPluginsList m_loaded_plugins;
    
    void
    InstanceInitialize ();
    
private:

    // Use Debugger::CreateInstance() to get a shared pointer to a new
    // debugger object
    Debugger (lldb::LogOutputCallback m_log_callback, void *baton);

    DISALLOW_COPY_AND_ASSIGN (Debugger);
    
};

} // namespace lldb_private

#endif  // #if defined(__cplusplus)
#endif  // liblldb_Debugger_h_
