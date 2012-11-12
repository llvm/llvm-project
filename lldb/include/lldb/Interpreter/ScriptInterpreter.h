//===-- ScriptInterpreter.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ScriptInterpreter_h_
#define liblldb_ScriptInterpreter_h_

#include "lldb/lldb-private.h"

#include "lldb/Core/Broadcaster.h"
#include "lldb/Core/Error.h"

#include "lldb/Utility/PseudoTerminal.h"


namespace lldb_private {

class ScriptInterpreterObject
{
public:
    ScriptInterpreterObject() :
    m_object(NULL)
    {}
    
    ScriptInterpreterObject(void* obj) :
    m_object(obj)
    {}
    
    ScriptInterpreterObject(const ScriptInterpreterObject& rhs)
    : m_object(rhs.m_object)
    {}
    
    virtual void*
    GetObject()
    {
        return m_object;
    }
    
    ScriptInterpreterObject&
    operator = (const ScriptInterpreterObject& rhs)
    {
        if (this != &rhs)
            m_object = rhs.m_object;
        return *this;
    }
        
    virtual
    ~ScriptInterpreterObject()
    {}
    
protected:
    void* m_object;
};

class ScriptInterpreter
{
public:

    typedef void (*SWIGInitCallback) (void);

    typedef bool (*SWIGBreakpointCallbackFunction) (const char *python_function_name,
                                                    const char *session_dictionary_name,
                                                    const lldb::StackFrameSP& frame_sp,
                                                    const lldb::BreakpointLocationSP &bp_loc_sp);
    
    typedef bool (*SWIGWatchpointCallbackFunction) (const char *python_function_name,
                                                    const char *session_dictionary_name,
                                                    const lldb::StackFrameSP& frame_sp,
                                                    const lldb::WatchpointSP &wp_sp);
    
    typedef bool (*SWIGPythonTypeScriptCallbackFunction) (const char *python_function_name,
                                                          void *session_dictionary,
                                                          const lldb::ValueObjectSP& valobj_sp,
                                                          void** pyfunct_wrapper,
                                                          std::string& retval);
    
    typedef void* (*SWIGPythonCreateSyntheticProvider) (const std::string python_class_name,
                                                        const char *session_dictionary_name,
                                                        const lldb::ValueObjectSP& valobj_sp);

    typedef void* (*SWIGPythonCreateOSPlugin) (const std::string python_class_name,
                                               const char *session_dictionary_name,
                                               const lldb::ProcessSP& process_sp);
    
    typedef uint32_t       (*SWIGPythonCalculateNumChildren)                   (void *implementor);
    typedef void*          (*SWIGPythonGetChildAtIndex)                        (void *implementor, uint32_t idx);
    typedef int            (*SWIGPythonGetIndexOfChildWithName)                (void *implementor, const char* child_name);
    typedef void*          (*SWIGPythonCastPyObjectToSBValue)                  (void* data);
    typedef bool           (*SWIGPythonUpdateSynthProviderInstance)            (void* data);
    typedef bool           (*SWIGPythonMightHaveChildrenSynthProviderInstance) (void* data);

    
    typedef bool           (*SWIGPythonCallCommand)                 (const char *python_function_name,
                                                                     const char *session_dictionary_name,
                                                                     lldb::DebuggerSP& debugger,
                                                                     const char* args,
                                                                     std::string& err_msg,
                                                                     lldb_private::CommandReturnObject& cmd_retobj);
    
    typedef bool           (*SWIGPythonCallModuleInit)              (const std::string python_module_name,
                                                                     const char *session_dictionary_name,
                                                                     lldb::DebuggerSP& debugger);

    typedef enum
    {
        eScriptReturnTypeCharPtr,
        eScriptReturnTypeBool,
        eScriptReturnTypeShortInt,
        eScriptReturnTypeShortIntUnsigned,
        eScriptReturnTypeInt,
        eScriptReturnTypeIntUnsigned,
        eScriptReturnTypeLongInt,
        eScriptReturnTypeLongIntUnsigned,
        eScriptReturnTypeLongLong,
        eScriptReturnTypeLongLongUnsigned,
        eScriptReturnTypeFloat,
        eScriptReturnTypeDouble,
        eScriptReturnTypeChar,
        eScriptReturnTypeCharStrOrNone
    } ScriptReturnType;
    
    ScriptInterpreter (CommandInterpreter &interpreter, lldb::ScriptLanguage script_lang);

    virtual ~ScriptInterpreter ();

    struct ExecuteScriptOptions
    {
    public:
        ExecuteScriptOptions () :
            m_enable_io(true),
            m_set_lldb_globals(true),
            m_maskout_errors(true)
        {
        }
        
        bool
        GetEnableIO () const
        {
            return m_enable_io;
        }
        
        bool
        GetSetLLDBGlobals () const
        {
            return m_set_lldb_globals;
        }
        
        bool
        GetMaskoutErrors () const
        {
            return m_maskout_errors;
        }
        
        ExecuteScriptOptions&
        SetEnableIO (bool enable)
        {
            m_enable_io = enable;
            return *this;
        }

        ExecuteScriptOptions&
        SetSetLLDBGlobals (bool set)
        {
            m_set_lldb_globals = set;
            return *this;
        }

        ExecuteScriptOptions&
        SetMaskoutErrors (bool maskout)
        {
            m_maskout_errors = maskout;
            return *this;
        }
        
    private:
        bool m_enable_io;
        bool m_set_lldb_globals;
        bool m_maskout_errors;
    };
    
    virtual bool
    ExecuteOneLine (const char *command,
                    CommandReturnObject *result,
                    const ExecuteScriptOptions &options = ExecuteScriptOptions()) = 0;

    virtual void
    ExecuteInterpreterLoop () = 0;

    virtual bool
    ExecuteOneLineWithReturn (const char *in_string,
                              ScriptReturnType return_type,
                              void *ret_value,
                              const ExecuteScriptOptions &options = ExecuteScriptOptions())
    {
        return true;
    }

    virtual bool
    ExecuteMultipleLines (const char *in_string,
                          const ExecuteScriptOptions &options = ExecuteScriptOptions())
    {
        return true;
    }

    virtual bool
    ExportFunctionDefinitionToInterpreter (StringList &function_def)
    {
        return false;
    }

    virtual bool
    GenerateBreakpointCommandCallbackData (StringList &input, std::string& output)
    {
        return false;
    }
    
    virtual bool
    GenerateWatchpointCommandCallbackData (StringList &input, std::string& output)
    {
        return false;
    }
    
    virtual bool
    GenerateTypeScriptFunction (const char* oneliner, std::string& output, void* name_token = NULL)
    {
        return false;
    }
    
    virtual bool
    GenerateTypeScriptFunction (StringList &input, std::string& output, void* name_token = NULL)
    {
        return false;
    }
    
    virtual bool
    GenerateScriptAliasFunction (StringList &input, std::string& output)
    {
        return false;
    }
    
    virtual bool
    GenerateTypeSynthClass (StringList &input, std::string& output, void* name_token = NULL)
    {
        return false;
    }
    
    virtual bool
    GenerateTypeSynthClass (const char* oneliner, std::string& output, void* name_token = NULL)
    {
        return false;
    }
    
    virtual lldb::ScriptInterpreterObjectSP
    CreateSyntheticScriptedProvider (std::string class_name,
                                     lldb::ValueObjectSP valobj)
    {
        return lldb::ScriptInterpreterObjectSP();
    }
    
    virtual lldb::ScriptInterpreterObjectSP
    CreateOSPlugin (std::string class_name,
                    lldb::ProcessSP process_sp)
    {
        return lldb::ScriptInterpreterObjectSP();
    }
    
    virtual lldb::ScriptInterpreterObjectSP
    OSPlugin_QueryForRegisterInfo (lldb::ScriptInterpreterObjectSP object)
    {
        return lldb::ScriptInterpreterObjectSP();
    }
    
    virtual lldb::ScriptInterpreterObjectSP
    OSPlugin_QueryForThreadsInfo (lldb::ScriptInterpreterObjectSP object)
    {
        return lldb::ScriptInterpreterObjectSP();
    }
    
    virtual lldb::ScriptInterpreterObjectSP
    OSPlugin_QueryForRegisterContextData (lldb::ScriptInterpreterObjectSP object,
                                          lldb::tid_t thread_id)
    {
        return lldb::ScriptInterpreterObjectSP();
    }
    
    virtual bool
    GenerateFunction(const char *signature, const StringList &input)
    {
        return false;
    }

    virtual void 
    CollectDataForBreakpointCommandCallback (BreakpointOptions *bp_options,
                                             CommandReturnObject &result);

    virtual void 
    CollectDataForWatchpointCommandCallback (WatchpointOptions *wp_options,
                                             CommandReturnObject &result);

    /// Set a one-liner as the callback for the breakpoint.
    virtual void 
    SetBreakpointCommandCallback (BreakpointOptions *bp_options,
                                  const char *oneliner)
    {
        return;
    }
    
    /// Set a one-liner as the callback for the watchpoint.
    virtual void 
    SetWatchpointCommandCallback (WatchpointOptions *wp_options,
                                  const char *oneliner)
    {
        return;
    }
    
    virtual bool
    GetScriptedSummary (const char *function_name,
                        lldb::ValueObjectSP valobj,
                        lldb::ScriptInterpreterObjectSP& callee_wrapper_sp,
                        std::string& retval)
    {
        return false;
    }
    
    virtual uint32_t
    CalculateNumChildren (const lldb::ScriptInterpreterObjectSP& implementor)
    {
        return 0;
    }
    
    virtual lldb::ValueObjectSP
    GetChildAtIndex (const lldb::ScriptInterpreterObjectSP& implementor, uint32_t idx)
    {
        return lldb::ValueObjectSP();
    }
    
    virtual int
    GetIndexOfChildWithName (const lldb::ScriptInterpreterObjectSP& implementor, const char* child_name)
    {
        return UINT32_MAX;
    }
    
    virtual bool
    UpdateSynthProviderInstance (const lldb::ScriptInterpreterObjectSP& implementor)
    {
        return false;
    }
    
    virtual bool
    MightHaveChildrenSynthProviderInstance (const lldb::ScriptInterpreterObjectSP& implementor)
    {
        return true;
    }
    
    virtual bool
    RunScriptBasedCommand (const char* impl_function,
                           const char* args,
                           ScriptedCommandSynchronicity synchronicity,
                           lldb_private::CommandReturnObject& cmd_retobj,
                           Error& error)
    {
        return false;
    }
    
    virtual bool
    GetDocumentationForItem (const char* item, std::string& dest)
    {
		dest.clear();
        return false;
    }

    virtual bool
    LoadScriptingModule (const char* filename,
                         bool can_reload,
                         bool init_session,
                         lldb_private::Error& error)
    {
        error.SetErrorString("loading unimplemented");
        return false;
    }

    virtual lldb::ScriptInterpreterObjectSP
    MakeScriptObject (void* object)
    {
        return lldb::ScriptInterpreterObjectSP(new ScriptInterpreterObject(object));
    }
    
    const char *
    GetScriptInterpreterPtyName ();

    int
    GetMasterFileDescriptor ();

	CommandInterpreter &
	GetCommandInterpreter ();

    static std::string
    LanguageToString (lldb::ScriptLanguage language);
    
    static void
    InitializeInterpreter (SWIGInitCallback python_swig_init_callback);

    static void
    TerminateInterpreter ();

    virtual void
    ResetOutputFileHandle (FILE *new_fh) { } //By default, do nothing.

protected:
    CommandInterpreter &m_interpreter;
    lldb::ScriptLanguage m_script_lang;
};

} // namespace lldb_private

#endif // #ifndef liblldb_ScriptInterpreter_h_
