//===-- ScriptInterpreterPython.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#ifndef liblldb_ScriptInterpreterPython_h_
#define liblldb_ScriptInterpreterPython_h_

#ifdef LLDB_DISABLE_PYTHON

// Python is disabled in this build

#else

#if defined (__APPLE__)
#include <Python/Python.h>
#else
#include <Python.h>
#endif

#include "lldb/lldb-private.h"
#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/Core/InputReader.h"
#include "lldb/Host/Terminal.h"

namespace lldb_private {
    
class ScriptInterpreterPython : public ScriptInterpreter
{
public:

    ScriptInterpreterPython (CommandInterpreter &interpreter);

    ~ScriptInterpreterPython ();

    bool
    ExecuteOneLine (const char *command,
                    CommandReturnObject *result,
                    const ExecuteScriptOptions &options = ExecuteScriptOptions());

    void
    ExecuteInterpreterLoop ();

    bool
    ExecuteOneLineWithReturn (const char *in_string, 
                              ScriptInterpreter::ScriptReturnType return_type,
                              void *ret_value,
                              const ExecuteScriptOptions &options = ExecuteScriptOptions());

    bool
    ExecuteMultipleLines (const char *in_string,
                          const ExecuteScriptOptions &options = ExecuteScriptOptions());

    bool
    ExportFunctionDefinitionToInterpreter (StringList &function_def);

    bool
    GenerateTypeScriptFunction (StringList &input, std::string& output, void* name_token = NULL);
    
    bool
    GenerateTypeSynthClass (StringList &input, std::string& output, void* name_token = NULL);
    
    bool
    GenerateTypeSynthClass (const char* oneliner, std::string& output, void* name_token = NULL);
    
    // use this if the function code is just a one-liner script
    bool
    GenerateTypeScriptFunction (const char* oneliner, std::string& output, void* name_token = NULL);
    
    virtual bool
    GenerateScriptAliasFunction (StringList &input, std::string& output);
    
    lldb::ScriptInterpreterObjectSP
    CreateSyntheticScriptedProvider (const char *class_name,
                                     lldb::ValueObjectSP valobj);
    
    virtual lldb::ScriptInterpreterObjectSP
    OSPlugin_CreatePluginObject (const char *class_name,
                                 lldb::ProcessSP process_sp);
    
    virtual lldb::ScriptInterpreterObjectSP
    OSPlugin_RegisterInfo (lldb::ScriptInterpreterObjectSP os_plugin_object_sp);
    
    virtual lldb::ScriptInterpreterObjectSP
    OSPlugin_ThreadsInfo (lldb::ScriptInterpreterObjectSP os_plugin_object_sp);
    
    virtual lldb::ScriptInterpreterObjectSP
    OSPlugin_RegisterContextData (lldb::ScriptInterpreterObjectSP os_plugin_object_sp,
                                  lldb::tid_t thread_id);
    
    virtual lldb::ScriptInterpreterObjectSP
    OSPlugin_CreateThread (lldb::ScriptInterpreterObjectSP os_plugin_object_sp,
                           lldb::tid_t tid,
                           lldb::addr_t context);
    
    virtual size_t
    CalculateNumChildren (const lldb::ScriptInterpreterObjectSP& implementor);
    
    virtual lldb::ValueObjectSP
    GetChildAtIndex (const lldb::ScriptInterpreterObjectSP& implementor, uint32_t idx);
    
    virtual int
    GetIndexOfChildWithName (const lldb::ScriptInterpreterObjectSP& implementor, const char* child_name);
    
    virtual bool
    UpdateSynthProviderInstance (const lldb::ScriptInterpreterObjectSP& implementor);
    
    virtual bool
    MightHaveChildrenSynthProviderInstance (const lldb::ScriptInterpreterObjectSP& implementor);
    
    virtual bool
    RunScriptBasedCommand(const char* impl_function,
                          const char* args,
                          ScriptedCommandSynchronicity synchronicity,
                          lldb_private::CommandReturnObject& cmd_retobj,
                          Error& error);
    
    bool
    GenerateFunction(const char *signature, const StringList &input);
    
    bool
    GenerateBreakpointCommandCallbackData (StringList &input, std::string& output);

    bool
    GenerateWatchpointCommandCallbackData (StringList &input, std::string& output);

    static size_t
    GenerateBreakpointOptionsCommandCallback (void *baton, 
                                              InputReader &reader, 
                                              lldb::InputReaderAction notification,
                                              const char *bytes, 
                                              size_t bytes_len);
        
    static size_t
    GenerateWatchpointOptionsCommandCallback (void *baton, 
                                              InputReader &reader, 
                                              lldb::InputReaderAction notification,
                                              const char *bytes, 
                                              size_t bytes_len);
        
    static bool
    BreakpointCallbackFunction (void *baton, 
                                StoppointCallbackContext *context, 
                                lldb::user_id_t break_id,
                                lldb::user_id_t break_loc_id);
    
    static bool
    WatchpointCallbackFunction (void *baton, 
                                StoppointCallbackContext *context, 
                                lldb::user_id_t watch_id);
    
    virtual bool
    GetScriptedSummary (const char *function_name,
                        lldb::ValueObjectSP valobj,
                        lldb::ScriptInterpreterObjectSP& callee_wrapper_sp,
                        std::string& retval);
    
    virtual bool
    GetDocumentationForItem (const char* item, std::string& dest);
    
    virtual bool
    CheckObjectExists (const char* name)
    {
        if (!name || !name[0])
            return false;
        std::string temp;
        return GetDocumentationForItem (name,temp);
    }
    
    virtual bool
    LoadScriptingModule (const char* filename,
                         bool can_reload,
                         bool init_session,
                         lldb_private::Error& error);
    
    virtual lldb::ScriptInterpreterObjectSP
    MakeScriptObject (void* object);
    
    virtual std::unique_ptr<ScriptInterpreterLocker>
    AcquireInterpreterLock ();
    
    void
    CollectDataForBreakpointCommandCallback (BreakpointOptions *bp_options,
                                             CommandReturnObject &result);

    void 
    CollectDataForWatchpointCommandCallback (WatchpointOptions *wp_options,
                                             CommandReturnObject &result);

    /// Set a Python one-liner as the callback for the breakpoint.
    void 
    SetBreakpointCommandCallback (BreakpointOptions *bp_options,
                                  const char *oneliner);

    /// Set a one-liner as the callback for the watchpoint.
    void 
    SetWatchpointCommandCallback (WatchpointOptions *wp_options,
                                  const char *oneliner);

    StringList
    ReadCommandInputFromUser (FILE *in_file);
    
    virtual void
    ResetOutputFileHandle (FILE *new_fh);
    
    static lldb::thread_result_t
    RunEmbeddedPythonInterpreter (lldb::thread_arg_t baton);

    static void
    InitializePrivate ();

    static void
    InitializeInterpreter (SWIGInitCallback python_swig_init_callback);

protected:

    bool
    EnterSession (bool init_lldb_globals);
    
    void
    LeaveSession ();
    
    void
    SaveTerminalState (int fd);

    void
    RestoreTerminalState ();
    
private:
    
    class SynchronicityHandler
    {
    private:
        lldb::DebuggerSP             m_debugger_sp;
        ScriptedCommandSynchronicity m_synch_wanted;
        bool                         m_old_asynch;
    public:
        SynchronicityHandler(lldb::DebuggerSP,
                             ScriptedCommandSynchronicity);
        ~SynchronicityHandler();
    };
    
    class ScriptInterpreterPythonObject : public ScriptInterpreterObject
    {
    public:
        ScriptInterpreterPythonObject() :
        ScriptInterpreterObject()
        {}
        
        ScriptInterpreterPythonObject(void* obj) :
        ScriptInterpreterObject(obj)
        {
            Py_XINCREF(m_object);
        }
        
        operator bool ()
        {
            return m_object && m_object != Py_None;
        }
        
        
        virtual
        ~ScriptInterpreterPythonObject()
        {
            Py_XDECREF(m_object);
            m_object = NULL;
        }
        private:
            DISALLOW_COPY_AND_ASSIGN (ScriptInterpreterPythonObject);
    };
    
	class Locker : public ScriptInterpreterLocker
	{
	public:
        
        enum OnEntry
        {
            AcquireLock         = 0x0001,
            InitSession         = 0x0002,
            InitGlobals         = 0x0004
        };
        
        enum OnLeave
        {
            FreeLock            = 0x0001,
            FreeAcquiredLock    = 0x0002,    // do not free the lock if we already held it when calling constructor
            TearDownSession     = 0x0004
        };
        
        Locker (ScriptInterpreterPython *py_interpreter = NULL,
                uint16_t on_entry = AcquireLock | InitSession,
                uint16_t on_leave = FreeLock | TearDownSession,
                FILE* wait_msg_handle = NULL);
        
    	~Locker ();

	private:
        
        bool
        DoAcquireLock ();
        
        bool
        DoInitSession (bool init_lldb_globals);
        
        bool
        DoFreeLock ();
        
        bool
        DoTearDownSession ();

        static void
        ReleasePythonLock ();
        
    	bool                     m_teardown_session;
    	ScriptInterpreterPython *m_python_interpreter;
    	FILE*                    m_tmp_fh;
        PyGILState_STATE         m_GILState;
	};
    
    class PythonInputReaderManager
    {
    public:
        PythonInputReaderManager (ScriptInterpreterPython *interpreter);
        
        operator bool()
        {
            return m_error;
        }
        
        ~PythonInputReaderManager();
        
    private:
        
        static size_t
        InputReaderCallback (void *baton,
                                           InputReader &reader,
                                           lldb::InputReaderAction notification,
                                           const char *bytes,
                                           size_t bytes_len);
        
        static lldb::thread_result_t
        RunPythonInputReader (lldb::thread_arg_t baton);
        
        ScriptInterpreterPython *m_interpreter;
        lldb::DebuggerSP m_debugger_sp;
        lldb::InputReaderSP m_reader_sp;
        bool m_error;
    };

    static size_t
    InputReaderCallback (void *baton, 
                         InputReader &reader, 
                         lldb::InputReaderAction notification,
                         const char *bytes, 
                         size_t bytes_len);


    lldb_utility::PseudoTerminal m_embedded_thread_pty;
    lldb_utility::PseudoTerminal m_embedded_python_pty;
    lldb::InputReaderSP m_embedded_thread_input_reader_sp;
    lldb::InputReaderSP m_embedded_python_input_reader_sp;
    FILE *m_dbg_stdout;
    PyObject *m_new_sysout;
    PyObject *m_old_sysout;
    PyObject *m_old_syserr;
    PyObject *m_run_one_line;
    std::string m_dictionary_name;
    TerminalState m_terminal_state;
    bool m_session_is_active;
    bool m_pty_slave_is_open;
    bool m_valid_session;
    PyThreadState *m_command_thread_state;
};
} // namespace lldb_private

#endif // #ifdef LLDB_DISABLE_PYTHON

#endif // #ifndef liblldb_ScriptInterpreterPython_h_
