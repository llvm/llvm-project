//===-- Debugger.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBDebugger.h"

#include "lldb/Core/Debugger.h"

#include <map>

#include "clang/AST/DeclCXX.h"
#include "clang/AST/Type.h"

#include "lldb/lldb-private.h"
#include "lldb/Core/ConnectionFileDescriptor.h"
#include "lldb/Core/DataVisualization.h"
#include "lldb/Core/FormatManager.h"
#include "lldb/Core/InputReader.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Core/State.h"
#include "lldb/Core/StreamAsynchronousIO.h"
#include "lldb/Core/StreamCallback.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/Timer.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Core/ValueObjectVariable.h"
#include "lldb/Host/DynamicLibrary.h"
#include "lldb/Host/Terminal.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/OptionValueSInt64.h"
#include "lldb/Interpreter/OptionValueString.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/TargetList.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/AnsiTerminal.h"

using namespace lldb;
using namespace lldb_private;


static uint32_t g_shared_debugger_refcount = 0;
static lldb::user_id_t g_unique_id = 1;

#pragma mark Static Functions

static Mutex &
GetDebuggerListMutex ()
{
    static Mutex g_mutex(Mutex::eMutexTypeRecursive);
    return g_mutex;
}

typedef std::vector<DebuggerSP> DebuggerList;

static DebuggerList &
GetDebuggerList()
{
    // hide the static debugger list inside a singleton accessor to avoid
    // global init contructors
    static DebuggerList g_list;
    return g_list;
}

OptionEnumValueElement
g_show_disassembly_enum_values[] =
{
    { Debugger::eStopDisassemblyTypeNever,    "never",     "Never show disassembly when displaying a stop context."},
    { Debugger::eStopDisassemblyTypeNoSource, "no-source", "Show disassembly when there is no source information, or the source file is missing when displaying a stop context."},
    { Debugger::eStopDisassemblyTypeAlways,   "always",    "Always show disassembly when displaying a stop context."},
    { 0, NULL, NULL }
};

OptionEnumValueElement
g_language_enumerators[] =
{
    { eScriptLanguageNone,      "none",     "Disable scripting languages."},
    { eScriptLanguagePython,    "python",   "Select python as the default scripting language."},
    { eScriptLanguageDefault,   "default",  "Select the lldb default as the default scripting language."},
    { 0, NULL, NULL }
};

#define MODULE_WITH_FUNC "{ ${module.file.basename}{`${function.name-with-args}${function.pc-offset}}}"
#define FILE_AND_LINE "{ at ${line.file.basename}:${line.number}}"

#define DEFAULT_THREAD_FORMAT "thread #${thread.index}: tid = ${thread.id}"\
    "{, ${frame.pc}}"\
    MODULE_WITH_FUNC\
    FILE_AND_LINE\
    "{, stop reason = ${thread.stop-reason}}"\
    "{\\nReturn value: ${thread.return-value}}"\
    "\\n"

#define DEFAULT_FRAME_FORMAT "frame #${frame.index}: ${frame.pc}"\
    MODULE_WITH_FUNC\
    FILE_AND_LINE\
    "\\n"



static PropertyDefinition
g_properties[] =
{
{   "auto-confirm",             OptionValue::eTypeBoolean, true, false, NULL, NULL, "If true all confirmation prompts will receive their default reply." },
{   "frame-format",             OptionValue::eTypeString , true, 0    , DEFAULT_FRAME_FORMAT, NULL, "The default frame format string to use when displaying stack frame information for threads." },
{   "notify-void",              OptionValue::eTypeBoolean, true, false, NULL, NULL, "Notify the user explicitly if an expression returns void (default: false)." },
{   "prompt",                   OptionValue::eTypeString , true, OptionValueString::eOptionEncodeCharacterEscapeSequences, "(lldb) ", NULL, "The debugger command line prompt displayed for the user." },
{   "script-lang",              OptionValue::eTypeEnum   , true, eScriptLanguagePython, NULL, g_language_enumerators, "The script language to be used for evaluating user-written scripts." },
{   "stop-disassembly-count",   OptionValue::eTypeSInt64 , true, 4    , NULL, NULL, "The number of disassembly lines to show when displaying a stopped context." },
{   "stop-disassembly-display", OptionValue::eTypeEnum   , true, Debugger::eStopDisassemblyTypeNoSource, NULL, g_show_disassembly_enum_values, "Control when to display disassembly when displaying a stopped context." },
{   "stop-line-count-after",    OptionValue::eTypeSInt64 , true, 3    , NULL, NULL, "The number of sources lines to display that come after the current source line when displaying a stopped context." },
{   "stop-line-count-before",   OptionValue::eTypeSInt64 , true, 3    , NULL, NULL, "The number of sources lines to display that come before the current source line when displaying a stopped context." },
{   "term-width",               OptionValue::eTypeSInt64 , true, 80   , NULL, NULL, "The maximum number of columns to use for displaying text." },
{   "thread-format",            OptionValue::eTypeString , true, 0    , DEFAULT_THREAD_FORMAT, NULL, "The default thread format string to use when displaying thread information." },
{   "use-external-editor",      OptionValue::eTypeBoolean, true, false, NULL, NULL, "Whether to use an external editor or not." },

    {   NULL,                       OptionValue::eTypeInvalid, true, 0    , NULL, NULL, NULL }
};

enum
{
    ePropertyAutoConfirm = 0,
    ePropertyFrameFormat,
    ePropertyNotiftVoid,
    ePropertyPrompt,
    ePropertyScriptLanguage,
    ePropertyStopDisassemblyCount,
    ePropertyStopDisassemblyDisplay,
    ePropertyStopLineCountAfter,
    ePropertyStopLineCountBefore,
    ePropertyTerminalWidth,
    ePropertyThreadFormat,
    ePropertyUseExternalEditor
};

//
//const char *
//Debugger::GetFrameFormat() const
//{
//    return m_properties_sp->GetFrameFormat();
//}
//const char *
//Debugger::GetThreadFormat() const
//{
//    return m_properties_sp->GetThreadFormat();
//}
//


Error
Debugger::SetPropertyValue (const ExecutionContext *exe_ctx,
                            VarSetOperationType op,
                            const char *property_path,
                            const char *value)
{
    Error error (Properties::SetPropertyValue (exe_ctx, op, property_path, value));
    if (error.Success())
    {
        if (strcmp(property_path, g_properties[ePropertyPrompt].name) == 0)
        {
            const char *new_prompt = GetPrompt();
            EventSP prompt_change_event_sp (new Event(CommandInterpreter::eBroadcastBitResetPrompt, new EventDataBytes (new_prompt)));
            GetCommandInterpreter().BroadcastEvent (prompt_change_event_sp);
        }
    }
    return error;
}

bool
Debugger::GetAutoConfirm () const
{
    const uint32_t idx = ePropertyAutoConfirm;
    return m_collection_sp->GetPropertyAtIndexAsBoolean (NULL, idx, g_properties[idx].default_uint_value != 0);
}

const char *
Debugger::GetFrameFormat() const
{
    const uint32_t idx = ePropertyFrameFormat;
    return m_collection_sp->GetPropertyAtIndexAsString (NULL, idx, g_properties[idx].default_cstr_value);
}

bool
Debugger::GetNotifyVoid () const
{
    const uint32_t idx = ePropertyNotiftVoid;
    return m_collection_sp->GetPropertyAtIndexAsBoolean (NULL, idx, g_properties[idx].default_uint_value != 0);
}

const char *
Debugger::GetPrompt() const
{
    const uint32_t idx = ePropertyPrompt;
    return m_collection_sp->GetPropertyAtIndexAsString (NULL, idx, g_properties[idx].default_cstr_value);
}

void
Debugger::SetPrompt(const char *p)
{
    const uint32_t idx = ePropertyPrompt;
    m_collection_sp->SetPropertyAtIndexAsString (NULL, idx, p);
    const char *new_prompt = GetPrompt();
    EventSP prompt_change_event_sp (new Event(CommandInterpreter::eBroadcastBitResetPrompt, new EventDataBytes (new_prompt)));;
    GetCommandInterpreter().BroadcastEvent (prompt_change_event_sp);
}

const char *
Debugger::GetThreadFormat() const
{
    const uint32_t idx = ePropertyThreadFormat;
    return m_collection_sp->GetPropertyAtIndexAsString (NULL, idx, g_properties[idx].default_cstr_value);
}

lldb::ScriptLanguage
Debugger::GetScriptLanguage() const
{
    const uint32_t idx = ePropertyScriptLanguage;
    return (lldb::ScriptLanguage)m_collection_sp->GetPropertyAtIndexAsEnumeration (NULL, idx, g_properties[idx].default_uint_value);
}

bool
Debugger::SetScriptLanguage (lldb::ScriptLanguage script_lang)
{
    const uint32_t idx = ePropertyScriptLanguage;
    return m_collection_sp->SetPropertyAtIndexAsEnumeration (NULL, idx, script_lang);
}

uint32_t
Debugger::GetTerminalWidth () const
{
    const uint32_t idx = ePropertyTerminalWidth;
    return m_collection_sp->GetPropertyAtIndexAsSInt64 (NULL, idx, g_properties[idx].default_uint_value);
}

bool
Debugger::SetTerminalWidth (uint32_t term_width)
{
    const uint32_t idx = ePropertyTerminalWidth;
    return m_collection_sp->SetPropertyAtIndexAsSInt64 (NULL, idx, term_width);
}

bool
Debugger::GetUseExternalEditor () const
{
    const uint32_t idx = ePropertyUseExternalEditor;
    return m_collection_sp->GetPropertyAtIndexAsBoolean (NULL, idx, g_properties[idx].default_uint_value != 0);
}

bool
Debugger::SetUseExternalEditor (bool b)
{
    const uint32_t idx = ePropertyUseExternalEditor;
    return m_collection_sp->SetPropertyAtIndexAsBoolean (NULL, idx, b);
}

uint32_t
Debugger::GetStopSourceLineCount (bool before) const
{
    const uint32_t idx = before ? ePropertyStopLineCountBefore : ePropertyStopLineCountAfter;
    return m_collection_sp->GetPropertyAtIndexAsSInt64 (NULL, idx, g_properties[idx].default_uint_value);
}

Debugger::StopDisassemblyType
Debugger::GetStopDisassemblyDisplay () const
{
    const uint32_t idx = ePropertyStopDisassemblyDisplay;
    return (Debugger::StopDisassemblyType)m_collection_sp->GetPropertyAtIndexAsEnumeration (NULL, idx, g_properties[idx].default_uint_value);
}

uint32_t
Debugger::GetDisassemblyLineCount () const
{
    const uint32_t idx = ePropertyStopDisassemblyCount;
    return m_collection_sp->GetPropertyAtIndexAsSInt64 (NULL, idx, g_properties[idx].default_uint_value);
}

#pragma mark Debugger

//const DebuggerPropertiesSP &
//Debugger::GetSettings() const
//{
//    return m_properties_sp;
//}
//

int
Debugger::TestDebuggerRefCount ()
{
    return g_shared_debugger_refcount;
}

void
Debugger::Initialize ()
{
    if (g_shared_debugger_refcount++ == 0)
        lldb_private::Initialize();
}

void
Debugger::Terminate ()
{
    if (g_shared_debugger_refcount > 0)
    {
        g_shared_debugger_refcount--;
        if (g_shared_debugger_refcount == 0)
        {
            lldb_private::WillTerminate();
            lldb_private::Terminate();

            // Clear our master list of debugger objects
            Mutex::Locker locker (GetDebuggerListMutex ());
            GetDebuggerList().clear();
        }
    }
}

void
Debugger::SettingsInitialize ()
{
    Target::SettingsInitialize ();
}

void
Debugger::SettingsTerminate ()
{
    Target::SettingsTerminate ();
}

bool
Debugger::LoadPlugin (const FileSpec& spec)
{
    lldb::DynamicLibrarySP dynlib_sp(new lldb_private::DynamicLibrary(spec));
    lldb::DebuggerSP debugger_sp(shared_from_this());
    lldb::SBDebugger debugger_sb(debugger_sp);
    // TODO: mangle this differently for your system - on OSX, the first underscore needs to be removed and the second one stays
    LLDBCommandPluginInit init_func = dynlib_sp->GetSymbol<LLDBCommandPluginInit>("_ZN4lldb16PluginInitializeENS_10SBDebuggerE");
    if (!init_func)
        return false;
    if (init_func(debugger_sb))
    {
        m_loaded_plugins.push_back(dynlib_sp);
        return true;
    }
    return false;
}

static FileSpec::EnumerateDirectoryResult
LoadPluginCallback
(
 void *baton,
 FileSpec::FileType file_type,
 const FileSpec &file_spec
 )
{
    Error error;
    
    static ConstString g_dylibext("dylib");
    
    if (!baton)
        return FileSpec::eEnumerateDirectoryResultQuit;
    
    Debugger *debugger = (Debugger*)baton;
    
    // If we have a regular file, a symbolic link or unknown file type, try
    // and process the file. We must handle unknown as sometimes the directory
    // enumeration might be enumerating a file system that doesn't have correct
    // file type information.
    if (file_type == FileSpec::eFileTypeRegular         ||
        file_type == FileSpec::eFileTypeSymbolicLink    ||
        file_type == FileSpec::eFileTypeUnknown          )
    {
        FileSpec plugin_file_spec (file_spec);
        plugin_file_spec.ResolvePath ();
        
        if (plugin_file_spec.GetFileNameExtension() != g_dylibext)
            return FileSpec::eEnumerateDirectoryResultNext;

        debugger->LoadPlugin (plugin_file_spec);
        
        return FileSpec::eEnumerateDirectoryResultNext;
    }
    
    else if (file_type == FileSpec::eFileTypeUnknown     ||
        file_type == FileSpec::eFileTypeDirectory   ||
        file_type == FileSpec::eFileTypeSymbolicLink )
    {
        // Try and recurse into anything that a directory or symbolic link.
        // We must also do this for unknown as sometimes the directory enumeration
        // might be enurating a file system that doesn't have correct file type
        // information.
        return FileSpec::eEnumerateDirectoryResultEnter;
    }
    
    return FileSpec::eEnumerateDirectoryResultNext;
}

void
Debugger::InstanceInitialize ()
{
    FileSpec dir_spec;
    const bool find_directories = true;
    const bool find_files = true;
    const bool find_other = true;
    char dir_path[PATH_MAX];
    if (Host::GetLLDBPath (ePathTypeLLDBSystemPlugins, dir_spec))
    {
        if (dir_spec.Exists() && dir_spec.GetPath(dir_path, sizeof(dir_path)))
        {
            FileSpec::EnumerateDirectory (dir_path,
                                          find_directories,
                                          find_files,
                                          find_other,
                                          LoadPluginCallback,
                                          this);
        }
    }
    
    if (Host::GetLLDBPath (ePathTypeLLDBUserPlugins, dir_spec))
    {
        if (dir_spec.Exists() && dir_spec.GetPath(dir_path, sizeof(dir_path)))
        {
            FileSpec::EnumerateDirectory (dir_path,
                                          find_directories,
                                          find_files,
                                          find_other,
                                          LoadPluginCallback,
                                          this);
        }
    }
    
    PluginManager::DebuggerInitialize (*this);
}

DebuggerSP
Debugger::CreateInstance (lldb::LogOutputCallback log_callback, void *baton)
{
    DebuggerSP debugger_sp (new Debugger(log_callback, baton));
    if (g_shared_debugger_refcount > 0)
    {
        Mutex::Locker locker (GetDebuggerListMutex ());
        GetDebuggerList().push_back(debugger_sp);
    }
    debugger_sp->InstanceInitialize ();
    return debugger_sp;
}

void
Debugger::Destroy (DebuggerSP &debugger_sp)
{
    if (debugger_sp.get() == NULL)
        return;
        
    debugger_sp->Clear();

    if (g_shared_debugger_refcount > 0)
    {
        Mutex::Locker locker (GetDebuggerListMutex ());
        DebuggerList &debugger_list = GetDebuggerList ();
        DebuggerList::iterator pos, end = debugger_list.end();
        for (pos = debugger_list.begin (); pos != end; ++pos)
        {
            if ((*pos).get() == debugger_sp.get())
            {
                debugger_list.erase (pos);
                return;
            }
        }
    }
}

DebuggerSP
Debugger::FindDebuggerWithInstanceName (const ConstString &instance_name)
{
    DebuggerSP debugger_sp;
    if (g_shared_debugger_refcount > 0)
    {
        Mutex::Locker locker (GetDebuggerListMutex ());
        DebuggerList &debugger_list = GetDebuggerList();
        DebuggerList::iterator pos, end = debugger_list.end();

        for (pos = debugger_list.begin(); pos != end; ++pos)
        {
            if ((*pos).get()->m_instance_name == instance_name)
            {
                debugger_sp = *pos;
                break;
            }
        }
    }
    return debugger_sp;
}

TargetSP
Debugger::FindTargetWithProcessID (lldb::pid_t pid)
{
    TargetSP target_sp;
    if (g_shared_debugger_refcount > 0)
    {
        Mutex::Locker locker (GetDebuggerListMutex ());
        DebuggerList &debugger_list = GetDebuggerList();
        DebuggerList::iterator pos, end = debugger_list.end();
        for (pos = debugger_list.begin(); pos != end; ++pos)
        {
            target_sp = (*pos)->GetTargetList().FindTargetWithProcessID (pid);
            if (target_sp)
                break;
        }
    }
    return target_sp;
}

TargetSP
Debugger::FindTargetWithProcess (Process *process)
{
    TargetSP target_sp;
    if (g_shared_debugger_refcount > 0)
    {
        Mutex::Locker locker (GetDebuggerListMutex ());
        DebuggerList &debugger_list = GetDebuggerList();
        DebuggerList::iterator pos, end = debugger_list.end();
        for (pos = debugger_list.begin(); pos != end; ++pos)
        {
            target_sp = (*pos)->GetTargetList().FindTargetWithProcess (process);
            if (target_sp)
                break;
        }
    }
    return target_sp;
}

Debugger::Debugger (lldb::LogOutputCallback log_callback, void *baton) :
    UserID (g_unique_id++),
    Properties(OptionValuePropertiesSP(new OptionValueProperties())), 
    m_input_comm("debugger.input"),
    m_input_file (),
    m_output_file (),
    m_error_file (),
    m_target_list (*this),
    m_platform_list (),
    m_listener ("lldb.Debugger"),
    m_source_manager(*this),
    m_source_file_cache(),
    m_command_interpreter_ap (new CommandInterpreter (*this, eScriptLanguageDefault, false)),
    m_input_reader_stack (),
    m_input_reader_data (),
    m_instance_name()
{
    char instance_cstr[256];
    snprintf(instance_cstr, sizeof(instance_cstr), "debugger_%d", (int)GetID());
    m_instance_name.SetCString(instance_cstr);
    if (log_callback)
        m_log_callback_stream_sp.reset (new StreamCallback (log_callback, baton));
    m_command_interpreter_ap->Initialize ();
    // Always add our default platform to the platform list
    PlatformSP default_platform_sp (Platform::GetDefaultPlatform());
    assert (default_platform_sp.get());
    m_platform_list.Append (default_platform_sp, true);
    
    m_collection_sp->Initialize (g_properties);
    m_collection_sp->AppendProperty (ConstString("target"),
                                     ConstString("Settings specify to debugging targets."),
                                     true,
                                     Target::GetGlobalProperties()->GetValueProperties());
    if (m_command_interpreter_ap.get())
    {
        m_collection_sp->AppendProperty (ConstString("interpreter"),
                                         ConstString("Settings specify to the debugger's command interpreter."),
                                         true,
                                         m_command_interpreter_ap->GetValueProperties());
    }
    OptionValueSInt64 *term_width = m_collection_sp->GetPropertyAtIndexAsOptionValueSInt64 (NULL, ePropertyTerminalWidth);
    term_width->SetMinimumValue(10);
    term_width->SetMaximumValue(1024);
}

Debugger::~Debugger ()
{
    Clear();
}

void
Debugger::Clear()
{
    CleanUpInputReaders();
    m_listener.Clear();
    int num_targets = m_target_list.GetNumTargets();
    for (int i = 0; i < num_targets; i++)
    {
        TargetSP target_sp (m_target_list.GetTargetAtIndex (i));
        if (target_sp)
        {
            ProcessSP process_sp (target_sp->GetProcessSP());
            if (process_sp)
            {
                if (process_sp->GetShouldDetach())
                    process_sp->Detach();
            }
            target_sp->Destroy();
        }
    }
    BroadcasterManager::Clear ();
    
    // Close the input file _before_ we close the input read communications class
    // as it does NOT own the input file, our m_input_file does.
    GetInputFile().Close ();
    // Now that we have closed m_input_file, we can now tell our input communication
    // class to close down. Its read thread should quickly exit after we close
    // the input file handle above.
    m_input_comm.Clear ();
}

bool
Debugger::GetCloseInputOnEOF () const
{
    return m_input_comm.GetCloseOnEOF();
}

void
Debugger::SetCloseInputOnEOF (bool b)
{
    m_input_comm.SetCloseOnEOF(b);
}

bool
Debugger::GetAsyncExecution ()
{
    return !m_command_interpreter_ap->GetSynchronous();
}

void
Debugger::SetAsyncExecution (bool async_execution)
{
    m_command_interpreter_ap->SetSynchronous (!async_execution);
}

    
void
Debugger::SetInputFileHandle (FILE *fh, bool tranfer_ownership)
{
    File &in_file = GetInputFile();
    in_file.SetStream (fh, tranfer_ownership);
    if (in_file.IsValid() == false)
        in_file.SetStream (stdin, true);

    // Disconnect from any old connection if we had one
    m_input_comm.Disconnect ();
    // Pass false as the second argument to ConnectionFileDescriptor below because
    // our "in_file" above will already take ownership if requested and we don't
    // want to objects trying to own and close a file descriptor.
    m_input_comm.SetConnection (new ConnectionFileDescriptor (in_file.GetDescriptor(), false));
    m_input_comm.SetReadThreadBytesReceivedCallback (Debugger::DispatchInputCallback, this);

    Error error;
    if (m_input_comm.StartReadThread (&error) == false)
    {
        File &err_file = GetErrorFile();

        err_file.Printf ("error: failed to main input read thread: %s", error.AsCString() ? error.AsCString() : "unkown error");
        exit(1);
    }
}

void
Debugger::SetOutputFileHandle (FILE *fh, bool tranfer_ownership)
{
    File &out_file = GetOutputFile();
    out_file.SetStream (fh, tranfer_ownership);
    if (out_file.IsValid() == false)
        out_file.SetStream (stdout, false);
    
    // do not create the ScriptInterpreter just for setting the output file handle
    // as the constructor will know how to do the right thing on its own
    const bool can_create = false;
    ScriptInterpreter* script_interpreter = GetCommandInterpreter().GetScriptInterpreter(can_create);
    if (script_interpreter)
        script_interpreter->ResetOutputFileHandle (fh);
}

void
Debugger::SetErrorFileHandle (FILE *fh, bool tranfer_ownership)
{
    File &err_file = GetErrorFile();
    err_file.SetStream (fh, tranfer_ownership);
    if (err_file.IsValid() == false)
        err_file.SetStream (stderr, false);
}

ExecutionContext
Debugger::GetSelectedExecutionContext ()
{
    ExecutionContext exe_ctx;
    TargetSP target_sp(GetSelectedTarget());
    exe_ctx.SetTargetSP (target_sp);
    
    if (target_sp)
    {
        ProcessSP process_sp (target_sp->GetProcessSP());
        exe_ctx.SetProcessSP (process_sp);
        if (process_sp && process_sp->IsRunning() == false)
        {
            ThreadSP thread_sp (process_sp->GetThreadList().GetSelectedThread());
            if (thread_sp)
            {
                exe_ctx.SetThreadSP (thread_sp);
                exe_ctx.SetFrameSP (thread_sp->GetSelectedFrame());
                if (exe_ctx.GetFramePtr() == NULL)
                    exe_ctx.SetFrameSP (thread_sp->GetStackFrameAtIndex (0));
            }
        }
    }
    return exe_ctx;

}

InputReaderSP 
Debugger::GetCurrentInputReader ()
{
    InputReaderSP reader_sp;
    
    if (!m_input_reader_stack.IsEmpty())
    {
        // Clear any finished readers from the stack
        while (CheckIfTopInputReaderIsDone()) ;
        
        if (!m_input_reader_stack.IsEmpty())
            reader_sp = m_input_reader_stack.Top();
    }
    
    return reader_sp;
}

void
Debugger::DispatchInputCallback (void *baton, const void *bytes, size_t bytes_len)
{
    if (bytes_len > 0)
        ((Debugger *)baton)->DispatchInput ((char *)bytes, bytes_len);
    else
        ((Debugger *)baton)->DispatchInputEndOfFile ();
}   


void
Debugger::DispatchInput (const char *bytes, size_t bytes_len)
{
    if (bytes == NULL || bytes_len == 0)
        return;

    WriteToDefaultReader (bytes, bytes_len);
}

void
Debugger::DispatchInputInterrupt ()
{
    m_input_reader_data.clear();
    
    InputReaderSP reader_sp (GetCurrentInputReader ());
    if (reader_sp)
    {
        reader_sp->Notify (eInputReaderInterrupt);
        
        // If notifying the reader of the interrupt finished the reader, we should pop it off the stack.
        while (CheckIfTopInputReaderIsDone ()) ;
    }
}

void
Debugger::DispatchInputEndOfFile ()
{
    m_input_reader_data.clear();
    
    InputReaderSP reader_sp (GetCurrentInputReader ());
    if (reader_sp)
    {
        reader_sp->Notify (eInputReaderEndOfFile);
        
        // If notifying the reader of the end-of-file finished the reader, we should pop it off the stack.
        while (CheckIfTopInputReaderIsDone ()) ;
    }
}

void
Debugger::CleanUpInputReaders ()
{
    m_input_reader_data.clear();
    
    // The bottom input reader should be the main debugger input reader.  We do not want to close that one here.
    while (m_input_reader_stack.GetSize() > 1)
    {
        InputReaderSP reader_sp (GetCurrentInputReader ());
        if (reader_sp)
        {
            reader_sp->Notify (eInputReaderEndOfFile);
            reader_sp->SetIsDone (true);
        }
    }
}

void
Debugger::NotifyTopInputReader (InputReaderAction notification)
{
    InputReaderSP reader_sp (GetCurrentInputReader());
    if (reader_sp)
	{
        reader_sp->Notify (notification);

        // Flush out any input readers that are done.
        while (CheckIfTopInputReaderIsDone ())
            /* Do nothing. */;
    }
}

bool
Debugger::InputReaderIsTopReader (const InputReaderSP& reader_sp)
{
    InputReaderSP top_reader_sp (GetCurrentInputReader());

    return (reader_sp.get() == top_reader_sp.get());
}
    

void
Debugger::WriteToDefaultReader (const char *bytes, size_t bytes_len)
{
    if (bytes && bytes_len)
        m_input_reader_data.append (bytes, bytes_len);

    if (m_input_reader_data.empty())
        return;

    while (!m_input_reader_stack.IsEmpty() && !m_input_reader_data.empty())
    {
        // Get the input reader from the top of the stack
        InputReaderSP reader_sp (GetCurrentInputReader ());
        if (!reader_sp)
            break;

        size_t bytes_handled = reader_sp->HandleRawBytes (m_input_reader_data.c_str(), 
                                                          m_input_reader_data.size());
        if (bytes_handled)
        {
            m_input_reader_data.erase (0, bytes_handled);
        }
        else
        {
            // No bytes were handled, we might not have reached our 
            // granularity, just return and wait for more data
            break;
        }
    }
    
    // Flush out any input readers that are done.
    while (CheckIfTopInputReaderIsDone ())
        /* Do nothing. */;

}

void
Debugger::PushInputReader (const InputReaderSP& reader_sp)
{
    if (!reader_sp)
        return;
 
    // Deactivate the old top reader
    InputReaderSP top_reader_sp (GetCurrentInputReader ());
    
    if (top_reader_sp)
        top_reader_sp->Notify (eInputReaderDeactivate);

    m_input_reader_stack.Push (reader_sp);
    reader_sp->Notify (eInputReaderActivate);
    ActivateInputReader (reader_sp);
}

bool
Debugger::PopInputReader (const InputReaderSP& pop_reader_sp)
{
    bool result = false;

    // The reader on the stop of the stack is done, so let the next
    // read on the stack referesh its prompt and if there is one...
    if (!m_input_reader_stack.IsEmpty())
    {
        // Cannot call GetCurrentInputReader here, as that would cause an infinite loop.
        InputReaderSP reader_sp(m_input_reader_stack.Top());
        
        if (!pop_reader_sp || pop_reader_sp.get() == reader_sp.get())
        {
            m_input_reader_stack.Pop ();
            reader_sp->Notify (eInputReaderDeactivate);
            reader_sp->Notify (eInputReaderDone);
            result = true;

            if (!m_input_reader_stack.IsEmpty())
            {
                reader_sp = m_input_reader_stack.Top();
                if (reader_sp)
                {
                    ActivateInputReader (reader_sp);
                    reader_sp->Notify (eInputReaderReactivate);
                }
            }
        }
    }
    return result;
}

bool
Debugger::CheckIfTopInputReaderIsDone ()
{
    bool result = false;
    if (!m_input_reader_stack.IsEmpty())
    {
        // Cannot call GetCurrentInputReader here, as that would cause an infinite loop.
        InputReaderSP reader_sp(m_input_reader_stack.Top());
        
        if (reader_sp && reader_sp->IsDone())
        {
            result = true;
            PopInputReader (reader_sp);
        }
    }
    return result;
}

void
Debugger::ActivateInputReader (const InputReaderSP &reader_sp)
{
    int input_fd = m_input_file.GetFile().GetDescriptor();

    if (input_fd >= 0)
    {
        Terminal tty(input_fd);
        
        tty.SetEcho(reader_sp->GetEcho());
                
        switch (reader_sp->GetGranularity())
        {
        case eInputReaderGranularityByte:
        case eInputReaderGranularityWord:
            tty.SetCanonical (false);
            break;

        case eInputReaderGranularityLine:
        case eInputReaderGranularityAll:
            tty.SetCanonical (true);
            break;

        default:
            break;
        }
    }
}

StreamSP
Debugger::GetAsyncOutputStream ()
{
    return StreamSP (new StreamAsynchronousIO (GetCommandInterpreter(),
                                               CommandInterpreter::eBroadcastBitAsynchronousOutputData));
}

StreamSP
Debugger::GetAsyncErrorStream ()
{
    return StreamSP (new StreamAsynchronousIO (GetCommandInterpreter(),
                                               CommandInterpreter::eBroadcastBitAsynchronousErrorData));
}    

uint32_t
Debugger::GetNumDebuggers()
{
    if (g_shared_debugger_refcount > 0)
    {
        Mutex::Locker locker (GetDebuggerListMutex ());
        return GetDebuggerList().size();
    }
    return 0;
}

lldb::DebuggerSP
Debugger::GetDebuggerAtIndex (uint32_t index)
{
    DebuggerSP debugger_sp;
    
    if (g_shared_debugger_refcount > 0)
    {
        Mutex::Locker locker (GetDebuggerListMutex ());
        DebuggerList &debugger_list = GetDebuggerList();
        
        if (index < debugger_list.size())
            debugger_sp = debugger_list[index];
    }

    return debugger_sp;
}

DebuggerSP
Debugger::FindDebuggerWithID (lldb::user_id_t id)
{
    DebuggerSP debugger_sp;

    if (g_shared_debugger_refcount > 0)
    {
        Mutex::Locker locker (GetDebuggerListMutex ());
        DebuggerList &debugger_list = GetDebuggerList();
        DebuggerList::iterator pos, end = debugger_list.end();
        for (pos = debugger_list.begin(); pos != end; ++pos)
        {
            if ((*pos).get()->GetID() == id)
            {
                debugger_sp = *pos;
                break;
            }
        }
    }
    return debugger_sp;
}

static void
TestPromptFormats (StackFrame *frame)
{
    if (frame == NULL)
        return;

    StreamString s;
    const char *prompt_format =         
    "{addr = '${addr}'\n}"
    "{process.id = '${process.id}'\n}"
    "{process.name = '${process.name}'\n}"
    "{process.file.basename = '${process.file.basename}'\n}"
    "{process.file.fullpath = '${process.file.fullpath}'\n}"
    "{thread.id = '${thread.id}'\n}"
    "{thread.index = '${thread.index}'\n}"
    "{thread.name = '${thread.name}'\n}"
    "{thread.queue = '${thread.queue}'\n}"
    "{thread.stop-reason = '${thread.stop-reason}'\n}"
    "{target.arch = '${target.arch}'\n}"
    "{module.file.basename = '${module.file.basename}'\n}"
    "{module.file.fullpath = '${module.file.fullpath}'\n}"
    "{file.basename = '${file.basename}'\n}"
    "{file.fullpath = '${file.fullpath}'\n}"
    "{frame.index = '${frame.index}'\n}"
    "{frame.pc = '${frame.pc}'\n}"
    "{frame.sp = '${frame.sp}'\n}"
    "{frame.fp = '${frame.fp}'\n}"
    "{frame.flags = '${frame.flags}'\n}"
    "{frame.reg.rdi = '${frame.reg.rdi}'\n}"
    "{frame.reg.rip = '${frame.reg.rip}'\n}"
    "{frame.reg.rsp = '${frame.reg.rsp}'\n}"
    "{frame.reg.rbp = '${frame.reg.rbp}'\n}"
    "{frame.reg.rflags = '${frame.reg.rflags}'\n}"
    "{frame.reg.xmm0 = '${frame.reg.xmm0}'\n}"
    "{frame.reg.carp = '${frame.reg.carp}'\n}"
    "{function.id = '${function.id}'\n}"
    "{function.name = '${function.name}'\n}"
    "{function.name-with-args = '${function.name-with-args}'\n}"
    "{function.addr-offset = '${function.addr-offset}'\n}"
    "{function.line-offset = '${function.line-offset}'\n}"
    "{function.pc-offset = '${function.pc-offset}'\n}"
    "{line.file.basename = '${line.file.basename}'\n}"
    "{line.file.fullpath = '${line.file.fullpath}'\n}"
    "{line.number = '${line.number}'\n}"
    "{line.start-addr = '${line.start-addr}'\n}"
    "{line.end-addr = '${line.end-addr}'\n}"
;

    SymbolContext sc (frame->GetSymbolContext(eSymbolContextEverything));
    ExecutionContext exe_ctx;
    frame->CalculateExecutionContext(exe_ctx);
    const char *end = NULL;
    if (Debugger::FormatPrompt (prompt_format, &sc, &exe_ctx, &sc.line_entry.range.GetBaseAddress(), s, &end))
    {
        printf("%s\n", s.GetData());
    }
    else
    {
        printf ("error: at '%s'\n", end);
        printf ("what we got: %s\n", s.GetData());
    }
}

static bool
ScanFormatDescriptor (const char* var_name_begin,
                      const char* var_name_end,
                      const char** var_name_final,
                      const char** percent_position,
                      Format* custom_format,
                      ValueObject::ValueObjectRepresentationStyle* val_obj_display)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_TYPES));
    *percent_position = ::strchr(var_name_begin,'%');
    if (!*percent_position || *percent_position > var_name_end)
    {
        if (log)
            log->Printf("[ScanFormatDescriptor] no format descriptor in string, skipping");
        *var_name_final = var_name_end;
    }
    else
    {
        *var_name_final = *percent_position;
        char* format_name = new char[var_name_end-*var_name_final]; format_name[var_name_end-*var_name_final-1] = '\0';
        memcpy(format_name, *var_name_final+1, var_name_end-*var_name_final-1);
        if (log)
            log->Printf("ScanFormatDescriptor] parsing %s as a format descriptor", format_name);
        if ( !FormatManager::GetFormatFromCString(format_name,
                                                  true,
                                                  *custom_format) )
        {
            if (log)
                log->Printf("ScanFormatDescriptor] %s is an unknown format", format_name);
            // if this is an @ sign, print ObjC description
            if (*format_name == '@')
                *val_obj_display = ValueObject::eValueObjectRepresentationStyleLanguageSpecific;
            // if this is a V, print the value using the default format
            else if (*format_name == 'V')
                *val_obj_display = ValueObject::eValueObjectRepresentationStyleValue;
            // if this is an L, print the location of the value
            else if (*format_name == 'L')
                *val_obj_display = ValueObject::eValueObjectRepresentationStyleLocation;
            // if this is an S, print the summary after all
            else if (*format_name == 'S')
                *val_obj_display = ValueObject::eValueObjectRepresentationStyleSummary;
            else if (*format_name == '#')
                *val_obj_display = ValueObject::eValueObjectRepresentationStyleChildrenCount;
            else if (*format_name == 'T')
                *val_obj_display = ValueObject::eValueObjectRepresentationStyleType;
            else if (log)
                log->Printf("ScanFormatDescriptor] %s is an error, leaving the previous value alone", format_name);
        }
        // a good custom format tells us to print the value using it
        else
        {
            if (log)
                log->Printf("ScanFormatDescriptor] will display value for this VO");
            *val_obj_display = ValueObject::eValueObjectRepresentationStyleValue;
        }
        delete format_name;
    }
    if (log)
        log->Printf("ScanFormatDescriptor] final format description outcome: custom_format = %d, val_obj_display = %d",
                    *custom_format,
                    *val_obj_display);
    return true;
}

static bool
ScanBracketedRange (const char* var_name_begin,
                    const char* var_name_end,
                    const char* var_name_final,
                    const char** open_bracket_position,
                    const char** separator_position,
                    const char** close_bracket_position,
                    const char** var_name_final_if_array_range,
                    int64_t* index_lower,
                    int64_t* index_higher)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_TYPES));
    *open_bracket_position = ::strchr(var_name_begin,'[');
    if (*open_bracket_position && *open_bracket_position < var_name_final)
    {
        *separator_position = ::strchr(*open_bracket_position,'-'); // might be NULL if this is a simple var[N] bitfield
        *close_bracket_position = ::strchr(*open_bracket_position,']');
        // as usual, we assume that [] will come before %
        //printf("trying to expand a []\n");
        *var_name_final_if_array_range = *open_bracket_position;
        if (*close_bracket_position - *open_bracket_position == 1)
        {
            if (log)
                log->Printf("[ScanBracketedRange] '[]' detected.. going from 0 to end of data");
            *index_lower = 0;
        }
        else if (*separator_position == NULL || *separator_position > var_name_end)
        {
            char *end = NULL;
            *index_lower = ::strtoul (*open_bracket_position+1, &end, 0);
            *index_higher = *index_lower;
            if (log)
                log->Printf("[ScanBracketedRange] [%lld] detected, high index is same", *index_lower);
        }
        else if (*close_bracket_position && *close_bracket_position < var_name_end)
        {
            char *end = NULL;
            *index_lower = ::strtoul (*open_bracket_position+1, &end, 0);
            *index_higher = ::strtoul (*separator_position+1, &end, 0);
            if (log)
                log->Printf("[ScanBracketedRange] [%lld-%lld] detected", *index_lower, *index_higher);
        }
        else
        {
            if (log)
                log->Printf("[ScanBracketedRange] expression is erroneous, cannot extract indices out of it");
            return false;
        }
        if (*index_lower > *index_higher && *index_higher > 0)
        {
            if (log)
                log->Printf("[ScanBracketedRange] swapping indices");
            int temp = *index_lower;
            *index_lower = *index_higher;
            *index_higher = temp;
        }
    }
    else if (log)
            log->Printf("[ScanBracketedRange] no bracketed range, skipping entirely");
    return true;
}

static ValueObjectSP
ExpandIndexedExpression (ValueObject* valobj,
                         uint32_t index,
                         StackFrame* frame,
                         bool deref_pointer)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_TYPES));
    const char* ptr_deref_format = "[%d]";
    std::auto_ptr<char> ptr_deref_buffer(new char[10]);
    ::sprintf(ptr_deref_buffer.get(), ptr_deref_format, index);
    if (log)
        log->Printf("[ExpandIndexedExpression] name to deref: %s",ptr_deref_buffer.get());
    const char* first_unparsed;
    ValueObject::GetValueForExpressionPathOptions options;
    ValueObject::ExpressionPathEndResultType final_value_type;
    ValueObject::ExpressionPathScanEndReason reason_to_stop;
    ValueObject::ExpressionPathAftermath what_next = (deref_pointer ? ValueObject::eExpressionPathAftermathDereference : ValueObject::eExpressionPathAftermathNothing);
    ValueObjectSP item = valobj->GetValueForExpressionPath (ptr_deref_buffer.get(),
                                                          &first_unparsed,
                                                          &reason_to_stop,
                                                          &final_value_type,
                                                          options,
                                                          &what_next);
    if (!item)
    {
        if (log)
            log->Printf("[ExpandIndexedExpression] ERROR: unparsed portion = %s, why stopping = %d,"
               " final_value_type %d",
               first_unparsed, reason_to_stop, final_value_type);
    }
    else
    {
        if (log)
            log->Printf("[ExpandIndexedExpression] ALL RIGHT: unparsed portion = %s, why stopping = %d,"
               " final_value_type %d",
               first_unparsed, reason_to_stop, final_value_type);
    }
    return item;
}

bool
Debugger::FormatPrompt 
(
    const char *format,
    const SymbolContext *sc,
    const ExecutionContext *exe_ctx,
    const Address *addr,
    Stream &s,
    const char **end,
    ValueObject* valobj
)
{
    ValueObject* realvalobj = NULL; // makes it super-easy to parse pointers
    bool success = true;
    const char *p;
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_TYPES));
    for (p = format; *p != '\0'; ++p)
    {
        if (realvalobj)
        {
            valobj = realvalobj;
            realvalobj = NULL;
        }
        size_t non_special_chars = ::strcspn (p, "${}\\");
        if (non_special_chars > 0)
        {
            if (success)
                s.Write (p, non_special_chars);
            p += non_special_chars;            
        }

        if (*p == '\0')
        {
            break;
        }
        else if (*p == '{')
        {
            // Start a new scope that must have everything it needs if it is to
            // to make it into the final output stream "s". If you want to make
            // a format that only prints out the function or symbol name if there
            // is one in the symbol context you can use:
            //      "{function =${function.name}}"
            // The first '{' starts a new scope that end with the matching '}' at
            // the end of the string. The contents "function =${function.name}"
            // will then be evaluated and only be output if there is a function
            // or symbol with a valid name. 
            StreamString sub_strm;

            ++p;  // Skip the '{'
            
            if (FormatPrompt (p, sc, exe_ctx, addr, sub_strm, &p, valobj))
            {
                // The stream had all it needed
                s.Write(sub_strm.GetData(), sub_strm.GetSize());
            }
            if (*p != '}')
            {
                success = false;
                break;
            }
        }
        else if (*p == '}')
        {
            // End of a enclosing scope
            break;
        }
        else if (*p == '$')
        {
            // We have a prompt variable to print
            ++p;
            if (*p == '{')
            {
                ++p;
                const char *var_name_begin = p;
                const char *var_name_end = ::strchr (p, '}');

                if (var_name_end && var_name_begin < var_name_end)
                {
                    // if we have already failed to parse, skip this variable
                    if (success)
                    {
                        const char *cstr = NULL;
                        Address format_addr;
                        bool calculate_format_addr_function_offset = false;
                        // Set reg_kind and reg_num to invalid values
                        RegisterKind reg_kind = kNumRegisterKinds; 
                        uint32_t reg_num = LLDB_INVALID_REGNUM;
                        FileSpec format_file_spec;
                        const RegisterInfo *reg_info = NULL;
                        RegisterContext *reg_ctx = NULL;
                        bool do_deref_pointer = false;
                        ValueObject::ExpressionPathScanEndReason reason_to_stop = ValueObject::eExpressionPathScanEndReasonEndOfString;
                        ValueObject::ExpressionPathEndResultType final_value_type = ValueObject::eExpressionPathEndResultTypePlain;
                        
                        // Each variable must set success to true below...
                        bool var_success = false;
                        switch (var_name_begin[0])
                        {
                        case '*':
                        case 'v':
                        case 's':
                            {
                                if (!valobj)
                                    break;
                                
                                if (log)
                                    log->Printf("[Debugger::FormatPrompt] initial string: %s",var_name_begin);
                                
                                // check for *var and *svar
                                if (*var_name_begin == '*')
                                {
                                    do_deref_pointer = true;
                                    var_name_begin++;
                                }
                                
                                if (log)
                                    log->Printf("[Debugger::FormatPrompt] initial string: %s",var_name_begin);
                                
                                if (*var_name_begin == 's')
                                {
                                    if (!valobj->IsSynthetic())
                                        valobj = valobj->GetSyntheticValue().get();
                                    if (!valobj)
                                        break;
                                    var_name_begin++;
                                }
                                
                                if (log)
                                    log->Printf("[Debugger::FormatPrompt] initial string: %s",var_name_begin);
                                
                                // should be a 'v' by now
                                if (*var_name_begin != 'v')
                                    break;
                                
                                if (log)
                                    log->Printf("[Debugger::FormatPrompt] initial string: %s",var_name_begin);
                                                                
                                ValueObject::ExpressionPathAftermath what_next = (do_deref_pointer ?
                                                                                  ValueObject::eExpressionPathAftermathDereference : ValueObject::eExpressionPathAftermathNothing);
                                ValueObject::GetValueForExpressionPathOptions options;
                                options.DontCheckDotVsArrowSyntax().DoAllowBitfieldSyntax().DoAllowFragileIVar().DoAllowSyntheticChildren();
                                ValueObject::ValueObjectRepresentationStyle val_obj_display = ValueObject::eValueObjectRepresentationStyleSummary;
                                ValueObject* target = NULL;
                                Format custom_format = eFormatInvalid;
                                const char* var_name_final = NULL;
                                const char* var_name_final_if_array_range = NULL;
                                const char* close_bracket_position = NULL;
                                int64_t index_lower = -1;
                                int64_t index_higher = -1;
                                bool is_array_range = false;
                                const char* first_unparsed;
                                bool was_plain_var = false;
                                bool was_var_format = false;
                                bool was_var_indexed = false;

                                if (!valobj) break;
                                // simplest case ${var}, just print valobj's value
                                if (::strncmp (var_name_begin, "var}", strlen("var}")) == 0)
                                {
                                    was_plain_var = true;
                                    target = valobj;
                                    val_obj_display = ValueObject::eValueObjectRepresentationStyleValue;
                                }
                                else if (::strncmp(var_name_begin,"var%",strlen("var%")) == 0)
                                {
                                    was_var_format = true;
                                    // this is a variable with some custom format applied to it
                                    const char* percent_position;
                                    target = valobj;
                                    val_obj_display = ValueObject::eValueObjectRepresentationStyleValue;
                                    ScanFormatDescriptor (var_name_begin,
                                                          var_name_end,
                                                          &var_name_final,
                                                          &percent_position,
                                                          &custom_format,
                                                          &val_obj_display);
                                }
                                    // this is ${var.something} or multiple .something nested
                                else if (::strncmp (var_name_begin, "var", strlen("var")) == 0)
                                {
                                    if (::strncmp(var_name_begin, "var[", strlen("var[")) == 0)
                                        was_var_indexed = true;
                                    const char* percent_position;
                                    ScanFormatDescriptor (var_name_begin,
                                                          var_name_end,
                                                          &var_name_final,
                                                          &percent_position,
                                                          &custom_format,
                                                          &val_obj_display);
                                    
                                    const char* open_bracket_position;
                                    const char* separator_position;
                                    ScanBracketedRange (var_name_begin,
                                                        var_name_end,
                                                        var_name_final,
                                                        &open_bracket_position,
                                                        &separator_position,
                                                        &close_bracket_position,
                                                        &var_name_final_if_array_range,
                                                        &index_lower,
                                                        &index_higher);
                                                                    
                                    Error error;
                                                                        
                                    std::auto_ptr<char> expr_path(new char[var_name_final-var_name_begin-1]);
                                    ::memset(expr_path.get(), 0, var_name_final-var_name_begin-1);
                                    memcpy(expr_path.get(), var_name_begin+3,var_name_final-var_name_begin-3);
                                                                        
                                    if (log)
                                        log->Printf("[Debugger::FormatPrompt] symbol to expand: %s",expr_path.get());
                                    
                                    target = valobj->GetValueForExpressionPath(expr_path.get(),
                                                                             &first_unparsed,
                                                                             &reason_to_stop,
                                                                             &final_value_type,
                                                                             options,
                                                                             &what_next).get();
                                    
                                    if (!target)
                                    {
                                        if (log)
                                            log->Printf("[Debugger::FormatPrompt] ERROR: unparsed portion = %s, why stopping = %d,"
                                               " final_value_type %d",
                                               first_unparsed, reason_to_stop, final_value_type);
                                        break;
                                    }
                                    else
                                    {
                                        if (log)
                                            log->Printf("[Debugger::FormatPrompt] ALL RIGHT: unparsed portion = %s, why stopping = %d,"
                                               " final_value_type %d",
                                               first_unparsed, reason_to_stop, final_value_type);
                                    }
                                }
                                else
                                    break;
                                
                                is_array_range = (final_value_type == ValueObject::eExpressionPathEndResultTypeBoundedRange ||
                                                  final_value_type == ValueObject::eExpressionPathEndResultTypeUnboundedRange);
                                
                                do_deref_pointer = (what_next == ValueObject::eExpressionPathAftermathDereference);

                                if (do_deref_pointer && !is_array_range)
                                {
                                    // I have not deref-ed yet, let's do it
                                    // this happens when we are not going through GetValueForVariableExpressionPath
                                    // to get to the target ValueObject
                                    Error error;
                                    target = target->Dereference(error).get();
                                    if (error.Fail())
                                    {
                                        if (log)
                                            log->Printf("[Debugger::FormatPrompt] ERROR: %s\n", error.AsCString("unknown")); \
                                        break;
                                    }
                                    do_deref_pointer = false;
                                }
                                
                                // <rdar://problem/11338654>
                                // we do not want to use the summary for a bitfield of type T:n
                                // if we were originally dealing with just a T - that would get
                                // us into an endless recursion
                                if (target->IsBitfield() && was_var_indexed)
                                {
                                    // TODO: check for a (T:n)-specific summary - we should still obey that
                                    StreamString bitfield_name;
                                    bitfield_name.Printf("%s:%d", target->GetTypeName().AsCString(), target->GetBitfieldBitSize());
                                    lldb::TypeNameSpecifierImplSP type_sp(new TypeNameSpecifierImpl(bitfield_name.GetData(),false));
                                    if (!DataVisualization::GetSummaryForType(type_sp))
                                        val_obj_display = ValueObject::eValueObjectRepresentationStyleValue;
                                }
                                
                                // TODO use flags for these
                                bool is_array = ClangASTContext::IsArrayType(target->GetClangType());
                                bool is_pointer = ClangASTContext::IsPointerType(target->GetClangType());
                                bool is_aggregate = ClangASTContext::IsAggregateType(target->GetClangType());
                                
                                if ((is_array || is_pointer) && (!is_array_range) && val_obj_display == ValueObject::eValueObjectRepresentationStyleValue) // this should be wrong, but there are some exceptions
                                {
                                    StreamString str_temp;
                                    if (log)
                                        log->Printf("[Debugger::FormatPrompt] I am into array || pointer && !range");
                                    
                                    if (target->HasSpecialPrintableRepresentation(val_obj_display,
                                                                                  custom_format))
                                    {
                                        // try to use the special cases
                                        var_success = target->DumpPrintableRepresentation(str_temp,
                                                                                          val_obj_display,
                                                                                          custom_format);
                                        if (log)
                                            log->Printf("[Debugger::FormatPrompt] special cases did%s match", var_success ? "" : "n't");
                                        
                                        // should not happen
                                        if (!var_success)
                                            s << "<invalid usage of pointer value as object>";
                                        else
                                            s << str_temp.GetData();
                                        var_success = true;
                                        break;
                                    }
                                    else
                                    {
                                        if (was_plain_var) // if ${var}
                                        {
                                            s << target->GetTypeName() << " @ " << target->GetLocationAsCString();
                                        }
                                        else if (is_pointer) // if pointer, value is the address stored
                                        {
                                            target->DumpPrintableRepresentation (s,
                                                                                 val_obj_display,
                                                                                 custom_format,
                                                                                 ValueObject::ePrintableRepresentationSpecialCasesDisable);
                                        }
                                        else
                                        {
                                            s << "<invalid usage of pointer value as object>";
                                        }
                                        var_success = true;
                                        break;
                                    }
                                }
                                
                                // if directly trying to print ${var}, and this is an aggregate, display a nice
                                // type @ location message
                                if (is_aggregate && was_plain_var)
                                {
                                    s << target->GetTypeName() << " @ " << target->GetLocationAsCString();
                                    var_success = true;
                                    break;
                                }
                                
                                // if directly trying to print ${var%V}, and this is an aggregate, do not let the user do it
                                if (is_aggregate && ((was_var_format && val_obj_display == ValueObject::eValueObjectRepresentationStyleValue)))
                                {
                                    s << "<invalid use of aggregate type>";
                                    var_success = true;
                                    break;
                                }
                                                                
                                if (!is_array_range)
                                {
                                    if (log)
                                        log->Printf("[Debugger::FormatPrompt] dumping ordinary printable output");
                                    var_success = target->DumpPrintableRepresentation(s,val_obj_display, custom_format);
                                }
                                else
                                {   
                                    if (log)
                                        log->Printf("[Debugger::FormatPrompt] checking if I can handle as array");
                                    if (!is_array && !is_pointer)
                                        break;
                                    if (log)
                                        log->Printf("[Debugger::FormatPrompt] handle as array");
                                    const char* special_directions = NULL;
                                    StreamString special_directions_writer;
                                    if (close_bracket_position && (var_name_end-close_bracket_position > 1))
                                    {
                                        ConstString additional_data;
                                        additional_data.SetCStringWithLength(close_bracket_position+1, var_name_end-close_bracket_position-1);
                                        special_directions_writer.Printf("${%svar%s}",
                                                                         do_deref_pointer ? "*" : "",
                                                                         additional_data.GetCString());
                                        special_directions = special_directions_writer.GetData();
                                    }
                                    
                                    // let us display items index_lower thru index_higher of this array
                                    s.PutChar('[');
                                    var_success = true;

                                    if (index_higher < 0)
                                        index_higher = valobj->GetNumChildren() - 1;
                                    
                                    uint32_t max_num_children = target->GetTargetSP()->GetMaximumNumberOfChildrenToDisplay();
                                    
                                    for (;index_lower<=index_higher;index_lower++)
                                    {
                                        ValueObject* item = ExpandIndexedExpression (target,
                                                                                     index_lower,
                                                                                     exe_ctx->GetFramePtr(),
                                                                                     false).get();
                                        
                                        if (!item)
                                        {
                                            if (log)
                                                log->Printf("[Debugger::FormatPrompt] ERROR in getting child item at index %lld", index_lower);
                                        }
                                        else
                                        {
                                            if (log)
                                                log->Printf("[Debugger::FormatPrompt] special_directions for child item: %s",special_directions);
                                        }

                                        if (!special_directions)
                                            var_success &= item->DumpPrintableRepresentation(s,val_obj_display, custom_format);
                                        else
                                            var_success &= FormatPrompt(special_directions, sc, exe_ctx, addr, s, NULL, item);
                                        
                                        if (--max_num_children == 0)
                                        {
                                            s.PutCString(", ...");
                                            break;
                                        }
                                        
                                        if (index_lower < index_higher)
                                            s.PutChar(',');
                                    }
                                    s.PutChar(']');
                                }
                            }
                            break;
                        case 'a':
                            if (::strncmp (var_name_begin, "addr}", strlen("addr}")) == 0)
                            {
                                if (addr && addr->IsValid())
                                {
                                    var_success = true;
                                    format_addr = *addr;
                                }
                            }
                            else if (::strncmp (var_name_begin, "ansi.", strlen("ansi.")) == 0)
                            {
                                var_success = true;
                                var_name_begin += strlen("ansi."); // Skip the "ansi."
                                if (::strncmp (var_name_begin, "fg.", strlen("fg.")) == 0)
                                {
                                    var_name_begin += strlen("fg."); // Skip the "fg."
                                    if (::strncmp (var_name_begin, "black}", strlen("black}")) == 0)
                                    {
                                        s.Printf ("%s%s%s", 
                                                  lldb_utility::ansi::k_escape_start, 
                                                  lldb_utility::ansi::k_fg_black,
                                                  lldb_utility::ansi::k_escape_end);
                                    }
                                    else if (::strncmp (var_name_begin, "red}", strlen("red}")) == 0)
                                    {
                                        s.Printf ("%s%s%s", 
                                                  lldb_utility::ansi::k_escape_start, 
                                                  lldb_utility::ansi::k_fg_red,
                                                  lldb_utility::ansi::k_escape_end);
                                    }
                                    else if (::strncmp (var_name_begin, "green}", strlen("green}")) == 0)
                                    {
                                        s.Printf ("%s%s%s", 
                                                  lldb_utility::ansi::k_escape_start, 
                                                  lldb_utility::ansi::k_fg_green,
                                                  lldb_utility::ansi::k_escape_end);
                                    }
                                    else if (::strncmp (var_name_begin, "yellow}", strlen("yellow}")) == 0)
                                    {
                                        s.Printf ("%s%s%s", 
                                                  lldb_utility::ansi::k_escape_start, 
                                                  lldb_utility::ansi::k_fg_yellow,
                                                  lldb_utility::ansi::k_escape_end);
                                    }
                                    else if (::strncmp (var_name_begin, "blue}", strlen("blue}")) == 0)
                                    {
                                        s.Printf ("%s%s%s", 
                                                  lldb_utility::ansi::k_escape_start, 
                                                  lldb_utility::ansi::k_fg_blue,
                                                  lldb_utility::ansi::k_escape_end);
                                    }
                                    else if (::strncmp (var_name_begin, "purple}", strlen("purple}")) == 0)
                                    {
                                        s.Printf ("%s%s%s", 
                                                  lldb_utility::ansi::k_escape_start, 
                                                  lldb_utility::ansi::k_fg_purple,
                                                  lldb_utility::ansi::k_escape_end);
                                    }
                                    else if (::strncmp (var_name_begin, "cyan}", strlen("cyan}")) == 0)
                                    {
                                        s.Printf ("%s%s%s", 
                                                  lldb_utility::ansi::k_escape_start, 
                                                  lldb_utility::ansi::k_fg_cyan,
                                                  lldb_utility::ansi::k_escape_end);
                                    }
                                    else if (::strncmp (var_name_begin, "white}", strlen("white}")) == 0)
                                    {
                                        s.Printf ("%s%s%s", 
                                                  lldb_utility::ansi::k_escape_start, 
                                                  lldb_utility::ansi::k_fg_white,
                                                  lldb_utility::ansi::k_escape_end);
                                    }
                                    else
                                    {
                                        var_success = false;
                                    }
                                }
                                else if (::strncmp (var_name_begin, "bg.", strlen("bg.")) == 0)
                                {
                                    var_name_begin += strlen("bg."); // Skip the "bg."
                                    if (::strncmp (var_name_begin, "black}", strlen("black}")) == 0)
                                    {
                                        s.Printf ("%s%s%s", 
                                                  lldb_utility::ansi::k_escape_start, 
                                                  lldb_utility::ansi::k_bg_black,
                                                  lldb_utility::ansi::k_escape_end);
                                    }
                                    else if (::strncmp (var_name_begin, "red}", strlen("red}")) == 0)
                                    {
                                        s.Printf ("%s%s%s", 
                                                  lldb_utility::ansi::k_escape_start, 
                                                  lldb_utility::ansi::k_bg_red,
                                                  lldb_utility::ansi::k_escape_end);
                                    }
                                    else if (::strncmp (var_name_begin, "green}", strlen("green}")) == 0)
                                    {
                                        s.Printf ("%s%s%s", 
                                                  lldb_utility::ansi::k_escape_start, 
                                                  lldb_utility::ansi::k_bg_green,
                                                  lldb_utility::ansi::k_escape_end);
                                    }
                                    else if (::strncmp (var_name_begin, "yellow}", strlen("yellow}")) == 0)
                                    {
                                        s.Printf ("%s%s%s", 
                                                  lldb_utility::ansi::k_escape_start, 
                                                  lldb_utility::ansi::k_bg_yellow,
                                                  lldb_utility::ansi::k_escape_end);
                                    }
                                    else if (::strncmp (var_name_begin, "blue}", strlen("blue}")) == 0)
                                    {
                                        s.Printf ("%s%s%s", 
                                                  lldb_utility::ansi::k_escape_start, 
                                                  lldb_utility::ansi::k_bg_blue,
                                                  lldb_utility::ansi::k_escape_end);
                                    }
                                    else if (::strncmp (var_name_begin, "purple}", strlen("purple}")) == 0)
                                    {
                                        s.Printf ("%s%s%s", 
                                                  lldb_utility::ansi::k_escape_start, 
                                                  lldb_utility::ansi::k_bg_purple,
                                                  lldb_utility::ansi::k_escape_end);
                                    }
                                    else if (::strncmp (var_name_begin, "cyan}", strlen("cyan}")) == 0)
                                    {
                                        s.Printf ("%s%s%s", 
                                                  lldb_utility::ansi::k_escape_start, 
                                                  lldb_utility::ansi::k_bg_cyan,
                                                  lldb_utility::ansi::k_escape_end);
                                    }
                                    else if (::strncmp (var_name_begin, "white}", strlen("white}")) == 0)
                                    {
                                        s.Printf ("%s%s%s", 
                                                  lldb_utility::ansi::k_escape_start, 
                                                  lldb_utility::ansi::k_bg_white,
                                                  lldb_utility::ansi::k_escape_end);
                                    }
                                    else
                                    {
                                        var_success = false;
                                    }
                                }
                                else if (::strncmp (var_name_begin, "normal}", strlen ("normal}")) == 0)
                                {
                                    s.Printf ("%s%s%s", 
                                              lldb_utility::ansi::k_escape_start, 
                                              lldb_utility::ansi::k_ctrl_normal,
                                              lldb_utility::ansi::k_escape_end);
                                }
                                else if (::strncmp (var_name_begin, "bold}", strlen("bold}")) == 0)
                                {
                                    s.Printf ("%s%s%s", 
                                              lldb_utility::ansi::k_escape_start, 
                                              lldb_utility::ansi::k_ctrl_bold,
                                              lldb_utility::ansi::k_escape_end);
                                }
                                else if (::strncmp (var_name_begin, "faint}", strlen("faint}")) == 0)
                                {
                                    s.Printf ("%s%s%s", 
                                              lldb_utility::ansi::k_escape_start, 
                                              lldb_utility::ansi::k_ctrl_faint,
                                              lldb_utility::ansi::k_escape_end);
                                }
                                else if (::strncmp (var_name_begin, "italic}", strlen("italic}")) == 0)
                                {
                                    s.Printf ("%s%s%s", 
                                              lldb_utility::ansi::k_escape_start, 
                                              lldb_utility::ansi::k_ctrl_italic,
                                              lldb_utility::ansi::k_escape_end);
                                }
                                else if (::strncmp (var_name_begin, "underline}", strlen("underline}")) == 0)
                                {
                                    s.Printf ("%s%s%s", 
                                              lldb_utility::ansi::k_escape_start, 
                                              lldb_utility::ansi::k_ctrl_underline,
                                              lldb_utility::ansi::k_escape_end);
                                }
                                else if (::strncmp (var_name_begin, "slow-blink}", strlen("slow-blink}")) == 0)
                                {
                                    s.Printf ("%s%s%s", 
                                              lldb_utility::ansi::k_escape_start, 
                                              lldb_utility::ansi::k_ctrl_slow_blink,
                                              lldb_utility::ansi::k_escape_end);
                                }
                                else if (::strncmp (var_name_begin, "fast-blink}", strlen("fast-blink}")) == 0)
                                {
                                    s.Printf ("%s%s%s", 
                                              lldb_utility::ansi::k_escape_start, 
                                              lldb_utility::ansi::k_ctrl_fast_blink,
                                              lldb_utility::ansi::k_escape_end);
                                }
                                else if (::strncmp (var_name_begin, "negative}", strlen("negative}")) == 0)
                                {
                                    s.Printf ("%s%s%s", 
                                              lldb_utility::ansi::k_escape_start, 
                                              lldb_utility::ansi::k_ctrl_negative,
                                              lldb_utility::ansi::k_escape_end);
                                }
                                else if (::strncmp (var_name_begin, "conceal}", strlen("conceal}")) == 0)
                                {
                                    s.Printf ("%s%s%s", 
                                              lldb_utility::ansi::k_escape_start, 
                                              lldb_utility::ansi::k_ctrl_conceal,
                                              lldb_utility::ansi::k_escape_end);

                                }
                                else if (::strncmp (var_name_begin, "crossed-out}", strlen("crossed-out}")) == 0)
                                {
                                    s.Printf ("%s%s%s", 
                                              lldb_utility::ansi::k_escape_start, 
                                              lldb_utility::ansi::k_ctrl_crossed_out,
                                              lldb_utility::ansi::k_escape_end);
                                }
                                else
                                {
                                    var_success = false;
                                }
                            }
                            break;

                        case 'p':
                            if (::strncmp (var_name_begin, "process.", strlen("process.")) == 0)
                            {
                                if (exe_ctx)
                                {
                                    Process *process = exe_ctx->GetProcessPtr();
                                    if (process)
                                    {
                                        var_name_begin += ::strlen ("process.");
                                        if (::strncmp (var_name_begin, "id}", strlen("id}")) == 0)
                                        {
                                            s.Printf("%llu", process->GetID());
                                            var_success = true;
                                        }
                                        else if ((::strncmp (var_name_begin, "name}", strlen("name}")) == 0) ||
                                                 (::strncmp (var_name_begin, "file.basename}", strlen("file.basename}")) == 0) ||
                                                 (::strncmp (var_name_begin, "file.fullpath}", strlen("file.fullpath}")) == 0))
                                        {
                                            Module *exe_module = process->GetTarget().GetExecutableModulePointer();
                                            if (exe_module)
                                            {
                                                if (var_name_begin[0] == 'n' || var_name_begin[5] == 'f')
                                                {
                                                    format_file_spec.GetFilename() = exe_module->GetFileSpec().GetFilename();
                                                    var_success = format_file_spec;
                                                }
                                                else
                                                {
                                                    format_file_spec = exe_module->GetFileSpec();
                                                    var_success = format_file_spec;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            break;
                        
                        case 't':
                            if (::strncmp (var_name_begin, "thread.", strlen("thread.")) == 0)
                            {
                                if (exe_ctx)
                                {
                                    Thread *thread = exe_ctx->GetThreadPtr();
                                    if (thread)
                                    {
                                        var_name_begin += ::strlen ("thread.");
                                        if (::strncmp (var_name_begin, "id}", strlen("id}")) == 0)
                                        {
                                            s.Printf("0x%4.4llx", thread->GetID());
                                            var_success = true;
                                        }
                                        else if (::strncmp (var_name_begin, "index}", strlen("index}")) == 0)
                                        {
                                            s.Printf("%u", thread->GetIndexID());
                                            var_success = true;
                                        }
                                        else if (::strncmp (var_name_begin, "name}", strlen("name}")) == 0)
                                        {
                                            cstr = thread->GetName();
                                            var_success = cstr && cstr[0];
                                            if (var_success)
                                                s.PutCString(cstr);
                                        }
                                        else if (::strncmp (var_name_begin, "queue}", strlen("queue}")) == 0)
                                        {
                                            cstr = thread->GetQueueName();
                                            var_success = cstr && cstr[0];
                                            if (var_success)
                                                s.PutCString(cstr);
                                        }
                                        else if (::strncmp (var_name_begin, "stop-reason}", strlen("stop-reason}")) == 0)
                                        {
                                            StopInfoSP stop_info_sp = thread->GetStopInfo ();
                                            if (stop_info_sp && stop_info_sp->IsValid())
                                            {
                                                cstr = stop_info_sp->GetDescription();
                                                if (cstr && cstr[0])
                                                {
                                                    s.PutCString(cstr);
                                                    var_success = true;
                                                }
                                            }
                                        }
                                        else if (::strncmp (var_name_begin, "return-value}", strlen("return-value}")) == 0)
                                        {
                                            StopInfoSP stop_info_sp = thread->GetStopInfo ();
                                            if (stop_info_sp && stop_info_sp->IsValid())
                                            {
                                                ValueObjectSP return_valobj_sp = StopInfo::GetReturnValueObject (stop_info_sp);
                                                if (return_valobj_sp)
                                                {
                                                    ValueObject::DumpValueObjectOptions dump_options;
                                                    ValueObject::DumpValueObject (s, return_valobj_sp.get(), dump_options);
                                                    var_success = true;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            else if (::strncmp (var_name_begin, "target.", strlen("target.")) == 0)
                            {
                                // TODO: hookup properties
//                                if (!target_properties_sp)
//                                {
//                                    Target *target = Target::GetTargetFromContexts (exe_ctx, sc);
//                                    if (target)
//                                        target_properties_sp = target->GetProperties();
//                                }
//
//                                if (target_properties_sp)
//                                {
//                                    var_name_begin += ::strlen ("target.");
//                                    const char *end_property = strchr(var_name_begin, '}');
//                                    if (end_property)
//                                    {
//                                        ConstString property_name(var_name_begin, end_property - var_name_begin);
//                                        std::string property_value (target_properties_sp->GetPropertyValue(property_name));
//                                        if (!property_value.empty())
//                                        {
//                                            s.PutCString (property_value.c_str());
//                                            var_success = true;
//                                        }
//                                    }
//                                }                                        
                                Target *target = Target::GetTargetFromContexts (exe_ctx, sc);
                                if (target)
                                {
                                    var_name_begin += ::strlen ("target.");
                                    if (::strncmp (var_name_begin, "arch}", strlen("arch}")) == 0)
                                    {
                                        ArchSpec arch (target->GetArchitecture ());
                                        if (arch.IsValid())
                                        {
                                            s.PutCString (arch.GetArchitectureName());
                                            var_success = true;
                                        }
                                    }
                                }
                            }
                            break;
                            
                            
                        case 'm':
                            if (::strncmp (var_name_begin, "module.", strlen("module.")) == 0)
                            {
                                if (sc && sc->module_sp.get())
                                {
                                    Module *module = sc->module_sp.get();
                                    var_name_begin += ::strlen ("module.");
                                    
                                    if (::strncmp (var_name_begin, "file.", strlen("file.")) == 0)
                                    {
                                        if (module->GetFileSpec())
                                        {
                                            var_name_begin += ::strlen ("file.");
                                            
                                            if (::strncmp (var_name_begin, "basename}", strlen("basename}")) == 0)
                                            {
                                                format_file_spec.GetFilename() = module->GetFileSpec().GetFilename();
                                                var_success = format_file_spec;
                                            }
                                            else if (::strncmp (var_name_begin, "fullpath}", strlen("fullpath}")) == 0)
                                            {
                                                format_file_spec = module->GetFileSpec();
                                                var_success = format_file_spec;
                                            }
                                        }
                                    }
                                }
                            }
                            break;
                            
                        
                        case 'f':
                            if (::strncmp (var_name_begin, "file.", strlen("file.")) == 0)
                            {
                                if (sc && sc->comp_unit != NULL)
                                {
                                    var_name_begin += ::strlen ("file.");
                                    
                                    if (::strncmp (var_name_begin, "basename}", strlen("basename}")) == 0)
                                    {
                                        format_file_spec.GetFilename() = sc->comp_unit->GetFilename();
                                        var_success = format_file_spec;
                                    }
                                    else if (::strncmp (var_name_begin, "fullpath}", strlen("fullpath}")) == 0)
                                    {
                                        format_file_spec = *sc->comp_unit;
                                        var_success = format_file_spec;
                                    }
                                }
                            }
                            else if (::strncmp (var_name_begin, "frame.", strlen("frame.")) == 0)
                            {
                                if (exe_ctx)
                                {
                                    StackFrame *frame = exe_ctx->GetFramePtr();
                                    if (frame)
                                    {
                                        var_name_begin += ::strlen ("frame.");
                                        if (::strncmp (var_name_begin, "index}", strlen("index}")) == 0)
                                        {
                                            s.Printf("%u", frame->GetFrameIndex());
                                            var_success = true;
                                        }
                                        else if (::strncmp (var_name_begin, "pc}", strlen("pc}")) == 0)
                                        {
                                            reg_kind = eRegisterKindGeneric;
                                            reg_num = LLDB_REGNUM_GENERIC_PC;
                                            var_success = true;
                                        }
                                        else if (::strncmp (var_name_begin, "sp}", strlen("sp}")) == 0)
                                        {
                                            reg_kind = eRegisterKindGeneric;
                                            reg_num = LLDB_REGNUM_GENERIC_SP;
                                            var_success = true;
                                        }
                                        else if (::strncmp (var_name_begin, "fp}", strlen("fp}")) == 0)
                                        {
                                            reg_kind = eRegisterKindGeneric;
                                            reg_num = LLDB_REGNUM_GENERIC_FP;
                                            var_success = true;
                                        }
                                        else if (::strncmp (var_name_begin, "flags}", strlen("flags}")) == 0)
                                        {
                                            reg_kind = eRegisterKindGeneric;
                                            reg_num = LLDB_REGNUM_GENERIC_FLAGS;
                                            var_success = true;
                                        }
                                        else if (::strncmp (var_name_begin, "reg.", strlen ("reg.")) == 0)
                                        {
                                            reg_ctx = frame->GetRegisterContext().get();
                                            if (reg_ctx)
                                            {
                                                var_name_begin += ::strlen ("reg.");
                                                if (var_name_begin < var_name_end)
                                                {
                                                    std::string reg_name (var_name_begin, var_name_end);
                                                    reg_info = reg_ctx->GetRegisterInfoByName (reg_name.c_str());
                                                    if (reg_info)
                                                        var_success = true;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            else if (::strncmp (var_name_begin, "function.", strlen("function.")) == 0)
                            {
                                if (sc && (sc->function != NULL || sc->symbol != NULL))
                                {
                                    var_name_begin += ::strlen ("function.");
                                    if (::strncmp (var_name_begin, "id}", strlen("id}")) == 0)
                                    {
                                        if (sc->function)
                                            s.Printf("function{0x%8.8llx}", sc->function->GetID());
                                        else
                                            s.Printf("symbol[%u]", sc->symbol->GetID());

                                        var_success = true;
                                    }
                                    else if (::strncmp (var_name_begin, "name}", strlen("name}")) == 0)
                                    {
                                        if (sc->function)
                                            cstr = sc->function->GetName().AsCString (NULL);
                                        else if (sc->symbol)
                                            cstr = sc->symbol->GetName().AsCString (NULL);
                                        if (cstr)
                                        {
                                            s.PutCString(cstr);
                                            
                                            if (sc->block)
                                            {
                                                Block *inline_block = sc->block->GetContainingInlinedBlock ();
                                                if (inline_block)
                                                {
                                                    const InlineFunctionInfo *inline_info = sc->block->GetInlinedFunctionInfo();
                                                    if (inline_info)
                                                    {
                                                        s.PutCString(" [inlined] ");
                                                        inline_info->GetName().Dump(&s);
                                                    }
                                                }
                                            }
                                            var_success = true;
                                        }
                                    }
                                    else if (::strncmp (var_name_begin, "name-with-args}", strlen("name-with-args}")) == 0)
                                    {
                                        // Print the function name with arguments in it

                                        if (sc->function)
                                        {
                                            var_success = true;
                                            ExecutionContextScope *exe_scope = exe_ctx ? exe_ctx->GetBestExecutionContextScope() : NULL;
                                            cstr = sc->function->GetName().AsCString (NULL);
                                            if (cstr)
                                            {
                                                const InlineFunctionInfo *inline_info = NULL;
                                                VariableListSP variable_list_sp;
                                                bool get_function_vars = true;
                                                if (sc->block)
                                                {
                                                    Block *inline_block = sc->block->GetContainingInlinedBlock ();

                                                    if (inline_block)
                                                    {
                                                        get_function_vars = false;
                                                        inline_info = sc->block->GetInlinedFunctionInfo();
                                                        if (inline_info)
                                                            variable_list_sp = inline_block->GetBlockVariableList (true);
                                                    }
                                                }
                                                
                                                if (get_function_vars)
                                                {
                                                    variable_list_sp = sc->function->GetBlock(true).GetBlockVariableList (true);
                                                }
                                                
                                                if (inline_info)
                                                {
                                                    s.PutCString (cstr);
                                                    s.PutCString (" [inlined] ");
                                                    cstr = inline_info->GetName().GetCString();
                                                }
                                                
                                                VariableList args;
                                                if (variable_list_sp)
                                                {
                                                    const size_t num_variables = variable_list_sp->GetSize();
                                                    for (size_t var_idx = 0; var_idx < num_variables; ++var_idx)
                                                    {
                                                        VariableSP var_sp (variable_list_sp->GetVariableAtIndex(var_idx));
                                                        if (var_sp->GetScope() == eValueTypeVariableArgument)
                                                            args.AddVariable (var_sp);
                                                    }

                                                }
                                                if (args.GetSize() > 0)
                                                {
                                                    const char *open_paren = strchr (cstr, '(');
                                                    const char *close_paren = NULL;
                                                    if (open_paren)
                                                        close_paren = strchr (open_paren, ')');
                                                    
                                                    if (open_paren)
                                                        s.Write(cstr, open_paren - cstr + 1);
                                                    else
                                                    {
                                                        s.PutCString (cstr);
                                                        s.PutChar ('(');
                                                    }
                                                    const size_t num_args = args.GetSize();
                                                    for (size_t arg_idx = 0; arg_idx < num_args; ++arg_idx)
                                                    {
                                                        VariableSP var_sp (args.GetVariableAtIndex (arg_idx));
                                                        ValueObjectSP var_value_sp (ValueObjectVariable::Create (exe_scope, var_sp));
                                                        const char *var_name = var_value_sp->GetName().GetCString();
                                                        const char *var_value = var_value_sp->GetValueAsCString();
                                                        if (var_value_sp->GetError().Success())
                                                        {
                                                            if (arg_idx > 0)
                                                                s.PutCString (", ");
                                                            s.Printf ("%s=%s", var_name, var_value);
                                                        }
                                                    }
                                                    
                                                    if (close_paren)
                                                        s.PutCString (close_paren);
                                                    else
                                                        s.PutChar(')');

                                                }
                                                else
                                                {
                                                    s.PutCString(cstr);
                                                }
                                            }
                                        }
                                        else if (sc->symbol)
                                        {
                                            cstr = sc->symbol->GetName().AsCString (NULL);
                                            if (cstr)
                                            {
                                                s.PutCString(cstr);
                                                var_success = true;
                                            }
                                        }
                                    }
                                    else if (::strncmp (var_name_begin, "addr-offset}", strlen("addr-offset}")) == 0)
                                    {
                                        var_success = addr != NULL;
                                        if (var_success)
                                        {
                                            format_addr = *addr;
                                            calculate_format_addr_function_offset = true;
                                        }
                                    }
                                    else if (::strncmp (var_name_begin, "line-offset}", strlen("line-offset}")) == 0)
                                    {
                                        var_success = sc->line_entry.range.GetBaseAddress().IsValid();
                                        if (var_success)
                                        {
                                            format_addr = sc->line_entry.range.GetBaseAddress();
                                            calculate_format_addr_function_offset = true;
                                        }
                                    }
                                    else if (::strncmp (var_name_begin, "pc-offset}", strlen("pc-offset}")) == 0)
                                    {
                                        StackFrame *frame = exe_ctx->GetFramePtr();
                                        var_success = frame != NULL;
                                        if (var_success)
                                        {
                                            format_addr = frame->GetFrameCodeAddress();
                                            calculate_format_addr_function_offset = true;
                                        }
                                    }
                                }
                            }
                            break;

                        case 'l':
                            if (::strncmp (var_name_begin, "line.", strlen("line.")) == 0)
                            {
                                if (sc && sc->line_entry.IsValid())
                                {
                                    var_name_begin += ::strlen ("line.");
                                    if (::strncmp (var_name_begin, "file.", strlen("file.")) == 0)
                                    {
                                        var_name_begin += ::strlen ("file.");
                                        
                                        if (::strncmp (var_name_begin, "basename}", strlen("basename}")) == 0)
                                        {
                                            format_file_spec.GetFilename() = sc->line_entry.file.GetFilename();
                                            var_success = format_file_spec;
                                        }
                                        else if (::strncmp (var_name_begin, "fullpath}", strlen("fullpath}")) == 0)
                                        {
                                            format_file_spec = sc->line_entry.file;
                                            var_success = format_file_spec;
                                        }
                                    }
                                    else if (::strncmp (var_name_begin, "number}", strlen("number}")) == 0)
                                    {
                                        var_success = true;
                                        s.Printf("%u", sc->line_entry.line);
                                    }
                                    else if ((::strncmp (var_name_begin, "start-addr}", strlen("start-addr}")) == 0) ||
                                             (::strncmp (var_name_begin, "end-addr}", strlen("end-addr}")) == 0))
                                    {
                                        var_success = sc && sc->line_entry.range.GetBaseAddress().IsValid();
                                        if (var_success)
                                        {
                                            format_addr = sc->line_entry.range.GetBaseAddress();
                                            if (var_name_begin[0] == 'e')
                                                format_addr.Slide (sc->line_entry.range.GetByteSize());
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        
                        if (var_success)
                        {
                            // If format addr is valid, then we need to print an address
                            if (reg_num != LLDB_INVALID_REGNUM)
                            {
                                StackFrame *frame = exe_ctx->GetFramePtr();
                                // We have a register value to display...
                                if (reg_num == LLDB_REGNUM_GENERIC_PC && reg_kind == eRegisterKindGeneric)
                                {
                                    format_addr = frame->GetFrameCodeAddress();
                                }
                                else
                                {
                                    if (reg_ctx == NULL)
                                        reg_ctx = frame->GetRegisterContext().get();

                                    if (reg_ctx)
                                    {
                                        if (reg_kind != kNumRegisterKinds)
                                            reg_num = reg_ctx->ConvertRegisterKindToRegisterNumber(reg_kind, reg_num);
                                        reg_info = reg_ctx->GetRegisterInfoAtIndex (reg_num);
                                        var_success = reg_info != NULL;
                                    }
                                }
                            }
                            
                            if (reg_info != NULL)
                            {
                                RegisterValue reg_value;
                                var_success = reg_ctx->ReadRegister (reg_info, reg_value);
                                if (var_success)
                                {
                                    reg_value.Dump(&s, reg_info, false, false, eFormatDefault);
                                }
                            }                            
                            
                            if (format_file_spec)
                            {
                                s << format_file_spec;
                            }

                            // If format addr is valid, then we need to print an address
                            if (format_addr.IsValid())
                            {
                                var_success = false;

                                if (calculate_format_addr_function_offset)
                                {
                                    Address func_addr;
                                    
                                    if (sc)
                                    {
                                        if (sc->function)
                                        {
                                            func_addr = sc->function->GetAddressRange().GetBaseAddress();
                                            if (sc->block)
                                            {
                                                // Check to make sure we aren't in an inline
                                                // function. If we are, use the inline block
                                                // range that contains "format_addr" since
                                                // blocks can be discontiguous.
                                                Block *inline_block = sc->block->GetContainingInlinedBlock ();
                                                AddressRange inline_range;
                                                if (inline_block && inline_block->GetRangeContainingAddress (format_addr, inline_range))
                                                    func_addr = inline_range.GetBaseAddress();
                                            }
                                        }
                                        else if (sc->symbol && sc->symbol->ValueIsAddress())
                                            func_addr = sc->symbol->GetAddress();
                                    }
                                    
                                    if (func_addr.IsValid())
                                    {
                                        if (func_addr.GetSection() == format_addr.GetSection())
                                        {
                                            addr_t func_file_addr = func_addr.GetFileAddress();
                                            addr_t addr_file_addr = format_addr.GetFileAddress();
                                            if (addr_file_addr > func_file_addr)
                                                s.Printf(" + %llu", addr_file_addr - func_file_addr);
                                            else if (addr_file_addr < func_file_addr)
                                                s.Printf(" - %llu", func_file_addr - addr_file_addr);
                                            var_success = true;
                                        }
                                        else
                                        {
                                            Target *target = Target::GetTargetFromContexts (exe_ctx, sc);
                                            if (target)
                                            {
                                                addr_t func_load_addr = func_addr.GetLoadAddress (target);
                                                addr_t addr_load_addr = format_addr.GetLoadAddress (target);
                                                if (addr_load_addr > func_load_addr)
                                                    s.Printf(" + %llu", addr_load_addr - func_load_addr);
                                                else if (addr_load_addr < func_load_addr)
                                                    s.Printf(" - %llu", func_load_addr - addr_load_addr);
                                                var_success = true;
                                            }
                                        }
                                    }
                                }
                                else
                                {
                                    Target *target = Target::GetTargetFromContexts (exe_ctx, sc);
                                    addr_t vaddr = LLDB_INVALID_ADDRESS;
                                    if (exe_ctx && !target->GetSectionLoadList().IsEmpty())
                                        vaddr = format_addr.GetLoadAddress (target);
                                    if (vaddr == LLDB_INVALID_ADDRESS)
                                        vaddr = format_addr.GetFileAddress ();

                                    if (vaddr != LLDB_INVALID_ADDRESS)
                                    {
                                        int addr_width = target->GetArchitecture().GetAddressByteSize() * 2;
                                        if (addr_width == 0)
                                            addr_width = 16;
                                        s.Printf("0x%*.*llx", addr_width, addr_width, vaddr);
                                        var_success = true;
                                    }
                                }
                            }
                        }

                        if (var_success == false)
                            success = false;
                    }
                    p = var_name_end;
                }
                else
                    break;
            }
            else
            {
                // We got a dollar sign with no '{' after it, it must just be a dollar sign
                s.PutChar(*p);
            }
        }
        else if (*p == '\\')
        {
            ++p; // skip the slash
            switch (*p)
            {
            case 'a': s.PutChar ('\a'); break;
            case 'b': s.PutChar ('\b'); break;
            case 'f': s.PutChar ('\f'); break;
            case 'n': s.PutChar ('\n'); break;
            case 'r': s.PutChar ('\r'); break;
            case 't': s.PutChar ('\t'); break;
            case 'v': s.PutChar ('\v'); break;
            case '\'': s.PutChar ('\''); break; 
            case '\\': s.PutChar ('\\'); break; 
            case '0':
                // 1 to 3 octal chars
                {
                    // Make a string that can hold onto the initial zero char,
                    // up to 3 octal digits, and a terminating NULL.
                    char oct_str[5] = { 0, 0, 0, 0, 0 };

                    int i;
                    for (i=0; (p[i] >= '0' && p[i] <= '7') && i<4; ++i)
                        oct_str[i] = p[i];

                    // We don't want to consume the last octal character since
                    // the main for loop will do this for us, so we advance p by
                    // one less than i (even if i is zero)
                    p += i - 1;
                    unsigned long octal_value = ::strtoul (oct_str, NULL, 8);
                    if (octal_value <= UINT8_MAX)
                    {
                        char octal_char = octal_value;
                        s.Write (&octal_char, 1);
                    }
                }
                break;

            case 'x':
                // hex number in the format 
                if (isxdigit(p[1]))
                {
                    ++p;    // Skip the 'x'

                    // Make a string that can hold onto two hex chars plus a
                    // NULL terminator
                    char hex_str[3] = { 0,0,0 };
                    hex_str[0] = *p;
                    if (isxdigit(p[1]))
                    {
                        ++p; // Skip the first of the two hex chars
                        hex_str[1] = *p;
                    }

                    unsigned long hex_value = strtoul (hex_str, NULL, 16);                    
                    if (hex_value <= UINT8_MAX)
                        s.PutChar (hex_value);
                }
                else
                {
                    s.PutChar('x');
                }
                break;
                
            default:
                // Just desensitize any other character by just printing what
                // came after the '\'
                s << *p;
                break;
            
            }

        }
    }
    if (end) 
        *end = p;
    return success;
}

void
Debugger::SetLoggingCallback (lldb::LogOutputCallback log_callback, void *baton)
{
    // For simplicity's sake, I am not going to deal with how to close down any
    // open logging streams, I just redirect everything from here on out to the
    // callback.
    m_log_callback_stream_sp.reset (new StreamCallback (log_callback, baton));
}

bool
Debugger::EnableLog (const char *channel, const char **categories, const char *log_file, uint32_t log_options, Stream &error_stream)
{
    Log::Callbacks log_callbacks;

    StreamSP log_stream_sp;
    if (m_log_callback_stream_sp)
    {
        log_stream_sp = m_log_callback_stream_sp;
        // For now when using the callback mode you always get thread & timestamp.
        log_options |= LLDB_LOG_OPTION_PREPEND_TIMESTAMP | LLDB_LOG_OPTION_PREPEND_THREAD_NAME;
    }
    else if (log_file == NULL || *log_file == '\0')
    {
        log_stream_sp.reset(new StreamFile(GetOutputFile().GetDescriptor(), false));
    }
    else
    {
        LogStreamMap::iterator pos = m_log_streams.find(log_file);
        if (pos == m_log_streams.end())
        {
            log_stream_sp.reset (new StreamFile (log_file));
            m_log_streams[log_file] = log_stream_sp;
        }
        else
            log_stream_sp = pos->second;
    }
    assert (log_stream_sp.get());
    
    if (log_options == 0)
        log_options = LLDB_LOG_OPTION_PREPEND_THREAD_NAME | LLDB_LOG_OPTION_THREADSAFE;
        
    if (Log::GetLogChannelCallbacks (channel, log_callbacks))
    {
        log_callbacks.enable (log_stream_sp, log_options, categories, &error_stream);
        return true;
    }
    else
    {
        LogChannelSP log_channel_sp (LogChannel::FindPlugin (channel));
        if (log_channel_sp)
        {
            if (log_channel_sp->Enable (log_stream_sp, log_options, &error_stream, categories))
            {
                return true;
            }
            else
            {
                error_stream.Printf ("Invalid log channel '%s'.\n", channel);
                return false;
            }
        }
        else
        {
            error_stream.Printf ("Invalid log channel '%s'.\n", channel);
            return false;
        }
    }
    return false;
}

