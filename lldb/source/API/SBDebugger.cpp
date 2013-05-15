//===-- SBDebugger.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include "lldb/API/SBDebugger.h"

#include "lldb/lldb-private.h"

#include "lldb/API/SBListener.h"
#include "lldb/API/SBBroadcaster.h"
#include "lldb/API/SBCommandInterpreter.h"
#include "lldb/API/SBCommandReturnObject.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBEvent.h"
#include "lldb/API/SBFrame.h"
#include "lldb/API/SBInputReader.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBSourceManager.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBStringList.h"
#include "lldb/API/SBTarget.h"
#include "lldb/API/SBThread.h"
#include "lldb/API/SBTypeCategory.h"
#include "lldb/API/SBTypeFormat.h"
#include "lldb/API/SBTypeFilter.h"
#include "lldb/API/SBTypeNameSpecifier.h"
#include "lldb/API/SBTypeSummary.h"
#include "lldb/API/SBTypeSynthetic.h"


#include "lldb/Core/Debugger.h"
#include "lldb/Core/State.h"
#include "lldb/DataFormatters/DataVisualization.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/OptionGroupPlatform.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/TargetList.h"

using namespace lldb;
using namespace lldb_private;

void
SBDebugger::Initialize ()
{
    Log *log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    if (log)
        log->Printf ("SBDebugger::Initialize ()");

    SBCommandInterpreter::InitializeSWIG ();

    Debugger::Initialize();
}

void
SBDebugger::Terminate ()
{
    Debugger::Terminate();
}

void
SBDebugger::Clear ()
{
    Log *log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    if (log)
        log->Printf ("SBDebugger(%p)::Clear ()", m_opaque_sp.get());
        
    if (m_opaque_sp)
        m_opaque_sp->CleanUpInputReaders ();

    m_opaque_sp.reset();
}

SBDebugger
SBDebugger::Create()
{
    return SBDebugger::Create(false, NULL, NULL);
}

SBDebugger
SBDebugger::Create(bool source_init_files)
{
    return SBDebugger::Create (source_init_files, NULL, NULL);
}

SBDebugger
SBDebugger::Create(bool source_init_files, lldb::LogOutputCallback callback, void *baton)

{
    Log *log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBDebugger debugger;
    debugger.reset(Debugger::CreateInstance(callback, baton));

    if (log)
    {
        SBStream sstr;
        debugger.GetDescription (sstr);
        log->Printf ("SBDebugger::Create () => SBDebugger(%p): %s", debugger.m_opaque_sp.get(), sstr.GetData());
    }

    SBCommandInterpreter interp = debugger.GetCommandInterpreter();
    if (source_init_files)
    {
        interp.get()->SkipLLDBInitFiles(false);
        interp.get()->SkipAppInitFiles (false);
        SBCommandReturnObject result;
        interp.SourceInitFileInHomeDirectory(result);
    }
    else
    {
        interp.get()->SkipLLDBInitFiles(true);
        interp.get()->SkipAppInitFiles (true);
    }
    return debugger;
}

void
SBDebugger::Destroy (SBDebugger &debugger)
{
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    
    if (log)
    {
        SBStream sstr;
        debugger.GetDescription (sstr);
        log->Printf ("SBDebugger::Destroy () => SBDebugger(%p): %s", debugger.m_opaque_sp.get(), sstr.GetData());
    }
    
    Debugger::Destroy (debugger.m_opaque_sp);
    
    if (debugger.m_opaque_sp.get() != NULL)
        debugger.m_opaque_sp.reset();
}

void
SBDebugger::MemoryPressureDetected ()
{
    // Since this function can be call asynchronously, we allow it to be
    // non-mandatory. We have seen deadlocks with this function when called
    // so we need to safeguard against this until we can determine what is
    // causing the deadlocks.
    Log *log (GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    
    const bool mandatory = false;
    if (log)
    {
        log->Printf ("SBDebugger::MemoryPressureDetected (), mandatory = %d", mandatory);
    }
    
    ModuleList::RemoveOrphanSharedModules(mandatory);
}

SBDebugger::SBDebugger () :
    m_opaque_sp ()
{
}

SBDebugger::SBDebugger(const lldb::DebuggerSP &debugger_sp) :
    m_opaque_sp(debugger_sp)
{
}

SBDebugger::SBDebugger(const SBDebugger &rhs) :
    m_opaque_sp (rhs.m_opaque_sp)
{
}

SBDebugger &
SBDebugger::operator = (const SBDebugger &rhs)
{
    if (this != &rhs)
    {
        m_opaque_sp = rhs.m_opaque_sp;
    }
    return *this;
}

SBDebugger::~SBDebugger ()
{
}

bool
SBDebugger::IsValid() const
{
    return m_opaque_sp.get() != NULL;
}


void
SBDebugger::SetAsync (bool b)
{
    if (m_opaque_sp)
        m_opaque_sp->SetAsyncExecution(b);
}

bool
SBDebugger::GetAsync ()
{
    if (m_opaque_sp)
        return m_opaque_sp->GetAsyncExecution();
    else
        return false;
}

void
SBDebugger::SkipLLDBInitFiles (bool b)
{
    if (m_opaque_sp)
        m_opaque_sp->GetCommandInterpreter().SkipLLDBInitFiles (b);
}

void
SBDebugger::SkipAppInitFiles (bool b)
{
    if (m_opaque_sp)
        m_opaque_sp->GetCommandInterpreter().SkipAppInitFiles (b);
}

// Shouldn't really be settable after initialization as this could cause lots of problems; don't want users
// trying to switch modes in the middle of a debugging session.
void
SBDebugger::SetInputFileHandle (FILE *fh, bool transfer_ownership)
{
    Log *log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    if (log)
        log->Printf ("SBDebugger(%p)::SetInputFileHandle (fh=%p, transfer_ownership=%i)", m_opaque_sp.get(),
                     fh, transfer_ownership);

    if (m_opaque_sp)
        m_opaque_sp->SetInputFileHandle (fh, transfer_ownership);
}

void
SBDebugger::SetOutputFileHandle (FILE *fh, bool transfer_ownership)
{
    Log *log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));


    if (log)
        log->Printf ("SBDebugger(%p)::SetOutputFileHandle (fh=%p, transfer_ownership=%i)", m_opaque_sp.get(),
                     fh, transfer_ownership);

    if (m_opaque_sp)
        m_opaque_sp->SetOutputFileHandle (fh, transfer_ownership);
}

void
SBDebugger::SetErrorFileHandle (FILE *fh, bool transfer_ownership)
{
    Log *log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));


    if (log)
        log->Printf ("SBDebugger(%p)::SetErrorFileHandle (fh=%p, transfer_ownership=%i)", m_opaque_sp.get(),
                     fh, transfer_ownership);

    if (m_opaque_sp)
        m_opaque_sp->SetErrorFileHandle (fh, transfer_ownership);
}

FILE *
SBDebugger::GetInputFileHandle ()
{
    if (m_opaque_sp)
        return m_opaque_sp->GetInputFile().GetStream();
    return NULL;
}

FILE *
SBDebugger::GetOutputFileHandle ()
{
    if (m_opaque_sp)
        return m_opaque_sp->GetOutputFile().GetStream();
    return NULL;
}

FILE *
SBDebugger::GetErrorFileHandle ()
{
    if (m_opaque_sp)
        return m_opaque_sp->GetErrorFile().GetStream();
    return NULL;
}

void
SBDebugger::SaveInputTerminalState()
{
    if (m_opaque_sp)
        m_opaque_sp->SaveInputTerminalState();
}

void
SBDebugger::RestoreInputTerminalState()
{
    if (m_opaque_sp)
        m_opaque_sp->RestoreInputTerminalState();

}
SBCommandInterpreter
SBDebugger::GetCommandInterpreter ()
{
    Log *log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBCommandInterpreter sb_interpreter;
    if (m_opaque_sp)
        sb_interpreter.reset (&m_opaque_sp->GetCommandInterpreter());

    if (log)
        log->Printf ("SBDebugger(%p)::GetCommandInterpreter () => SBCommandInterpreter(%p)", 
                     m_opaque_sp.get(), sb_interpreter.get());

    return sb_interpreter;
}

void
SBDebugger::HandleCommand (const char *command)
{
    if (m_opaque_sp)
    {
        TargetSP target_sp (m_opaque_sp->GetSelectedTarget());
        Mutex::Locker api_locker;
        if (target_sp)
            api_locker.Lock(target_sp->GetAPIMutex());

        SBCommandInterpreter sb_interpreter(GetCommandInterpreter ());
        SBCommandReturnObject result;

        sb_interpreter.HandleCommand (command, result, false);

        if (GetErrorFileHandle() != NULL)
            result.PutError (GetErrorFileHandle());
        if (GetOutputFileHandle() != NULL)
            result.PutOutput (GetOutputFileHandle());

        if (m_opaque_sp->GetAsyncExecution() == false)
        {
            SBProcess process(GetCommandInterpreter().GetProcess ());
            ProcessSP process_sp (process.GetSP());
            if (process_sp)
            {
                EventSP event_sp;
                Listener &lldb_listener = m_opaque_sp->GetListener();
                while (lldb_listener.GetNextEventForBroadcaster (process_sp.get(), event_sp))
                {
                    SBEvent event(event_sp);
                    HandleProcessEvent (process, event, GetOutputFileHandle(), GetErrorFileHandle());
                }
            }
        }
    }
}

SBListener
SBDebugger::GetListener ()
{
    Log *log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBListener sb_listener;
    if (m_opaque_sp)
        sb_listener.reset(&m_opaque_sp->GetListener(), false);

    if (log)
        log->Printf ("SBDebugger(%p)::GetListener () => SBListener(%p)", m_opaque_sp.get(),
                     sb_listener.get());

    return sb_listener;
}

void
SBDebugger::HandleProcessEvent (const SBProcess &process, const SBEvent &event, FILE *out, FILE *err)
{
    if (!process.IsValid())
        return;

    TargetSP target_sp (process.GetTarget().GetSP());
    if (!target_sp)
        return;

    const uint32_t event_type = event.GetType();
    char stdio_buffer[1024];
    size_t len;

    Mutex::Locker api_locker (target_sp->GetAPIMutex());
    
    if (event_type & (Process::eBroadcastBitSTDOUT | Process::eBroadcastBitStateChanged))
    {
        // Drain stdout when we stop just in case we have any bytes
        while ((len = process.GetSTDOUT (stdio_buffer, sizeof (stdio_buffer))) > 0)
            if (out != NULL)
                ::fwrite (stdio_buffer, 1, len, out);
    }
    
    if (event_type & (Process::eBroadcastBitSTDERR | Process::eBroadcastBitStateChanged))
    {
        // Drain stderr when we stop just in case we have any bytes
        while ((len = process.GetSTDERR (stdio_buffer, sizeof (stdio_buffer))) > 0)
            if (err != NULL)
                ::fwrite (stdio_buffer, 1, len, err);
    }
    
    if (event_type & Process::eBroadcastBitStateChanged)
    {
        StateType event_state = SBProcess::GetStateFromEvent (event);
        
        if (event_state == eStateInvalid)
            return;
        
        bool is_stopped = StateIsStoppedState (event_state);
        if (!is_stopped)
            process.ReportEventState (event, out);
    }
}

SBSourceManager
SBDebugger::GetSourceManager ()
{
    SBSourceManager sb_source_manager (*this);
    return sb_source_manager;
}


bool
SBDebugger::GetDefaultArchitecture (char *arch_name, size_t arch_name_len)
{
    if (arch_name && arch_name_len)
    {
        ArchSpec default_arch = Target::GetDefaultArchitecture ();

        if (default_arch.IsValid())
        {
            const std::string &triple_str = default_arch.GetTriple().str();
            if (!triple_str.empty())
                ::snprintf (arch_name, arch_name_len, "%s", triple_str.c_str());
            else
                ::snprintf (arch_name, arch_name_len, "%s", default_arch.GetArchitectureName());
            return true;
        }
    }
    if (arch_name && arch_name_len)
        arch_name[0] = '\0';
    return false;
}


bool
SBDebugger::SetDefaultArchitecture (const char *arch_name)
{
    if (arch_name)
    {
        ArchSpec arch (arch_name);
        if (arch.IsValid())
        {
            Target::SetDefaultArchitecture (arch);
            return true;
        }
    }
    return false;
}

ScriptLanguage
SBDebugger::GetScriptingLanguage (const char *script_language_name)
{

    return Args::StringToScriptLanguage (script_language_name,
                                         eScriptLanguageDefault,
                                         NULL);
}

const char *
SBDebugger::GetVersionString ()
{
    return GetVersion();
}

const char *
SBDebugger::StateAsCString (StateType state)
{
    return lldb_private::StateAsCString (state);
}

bool
SBDebugger::StateIsRunningState (StateType state)
{
    Log *log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    const bool result = lldb_private::StateIsRunningState (state);
    if (log)
        log->Printf ("SBDebugger::StateIsRunningState (state=%s) => %i", 
                     StateAsCString (state), result);

    return result;
}

bool
SBDebugger::StateIsStoppedState (StateType state)
{
    Log *log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    const bool result = lldb_private::StateIsStoppedState (state, false);
    if (log)
        log->Printf ("SBDebugger::StateIsStoppedState (state=%s) => %i", 
                     StateAsCString (state), result);

    return result;
}

lldb::SBTarget
SBDebugger::CreateTarget (const char *filename,
                          const char *target_triple,
                          const char *platform_name,
                          bool add_dependent_modules,
                          lldb::SBError& sb_error)
{
    SBTarget sb_target;
    TargetSP target_sp;
    if (m_opaque_sp)
    {
        sb_error.Clear();
        OptionGroupPlatform platform_options (false);
        platform_options.SetPlatformName (platform_name);
        
        sb_error.ref() = m_opaque_sp->GetTargetList().CreateTarget (*m_opaque_sp, 
                                                                    filename, 
                                                                    target_triple, 
                                                                    add_dependent_modules, 
                                                                    &platform_options,
                                                                    target_sp);
    
        if (sb_error.Success())
            sb_target.SetSP (target_sp);
    }
    else
    {
        sb_error.SetErrorString("invalid target");
    }
    
    Log *log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        log->Printf ("SBDebugger(%p)::CreateTarget (filename=\"%s\", triple=%s, platform_name=%s, add_dependent_modules=%u, error=%s) => SBTarget(%p)", 
                     m_opaque_sp.get(), 
                     filename, 
                     target_triple,
                     platform_name,
                     add_dependent_modules,
                     sb_error.GetCString(),
                     target_sp.get());
    }
    
    return sb_target;
}

SBTarget
SBDebugger::CreateTargetWithFileAndTargetTriple (const char *filename,
                                                 const char *target_triple)
{
    SBTarget sb_target;
    TargetSP target_sp;
    if (m_opaque_sp)
    {
        const bool add_dependent_modules = true;
        Error error (m_opaque_sp->GetTargetList().CreateTarget (*m_opaque_sp, 
                                                                filename, 
                                                                target_triple, 
                                                                add_dependent_modules, 
                                                                NULL,
                                                                target_sp));
        sb_target.SetSP (target_sp);
    }
    
    Log *log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        log->Printf ("SBDebugger(%p)::CreateTargetWithFileAndTargetTriple (filename=\"%s\", triple=%s) => SBTarget(%p)", 
                     m_opaque_sp.get(), filename, target_triple, target_sp.get());
    }

    return sb_target;
}

SBTarget
SBDebugger::CreateTargetWithFileAndArch (const char *filename, const char *arch_cstr)
{
    Log *log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBTarget sb_target;
    TargetSP target_sp;
    if (m_opaque_sp)
    {
        Error error;
        const bool add_dependent_modules = true;

        error = m_opaque_sp->GetTargetList().CreateTarget (*m_opaque_sp,
                                                           filename, 
                                                           arch_cstr, 
                                                           add_dependent_modules, 
                                                           NULL, 
                                                           target_sp);

        if (error.Success())
        {
            m_opaque_sp->GetTargetList().SetSelectedTarget (target_sp.get());
            sb_target.SetSP (target_sp);
        }
    }

    if (log)
    {
        log->Printf ("SBDebugger(%p)::CreateTargetWithFileAndArch (filename=\"%s\", arch=%s) => SBTarget(%p)", 
                     m_opaque_sp.get(), filename, arch_cstr, target_sp.get());
    }

    return sb_target;
}

SBTarget
SBDebugger::CreateTarget (const char *filename)
{
    SBTarget sb_target;
    TargetSP target_sp;
    if (m_opaque_sp)
    {
        ArchSpec arch = Target::GetDefaultArchitecture ();
        Error error;
        const bool add_dependent_modules = true;

        PlatformSP platform_sp(m_opaque_sp->GetPlatformList().GetSelectedPlatform());
        error = m_opaque_sp->GetTargetList().CreateTarget (*m_opaque_sp, 
                                                           filename, 
                                                           arch, 
                                                           add_dependent_modules, 
                                                           platform_sp,
                                                           target_sp);

        if (error.Success())
        {
            m_opaque_sp->GetTargetList().SetSelectedTarget (target_sp.get());
            sb_target.SetSP (target_sp);
        }
    }
    Log *log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        log->Printf ("SBDebugger(%p)::CreateTarget (filename=\"%s\") => SBTarget(%p)", 
                     m_opaque_sp.get(), filename, target_sp.get());
    }
    return sb_target;
}

bool
SBDebugger::DeleteTarget (lldb::SBTarget &target)
{
    bool result = false;
    if (m_opaque_sp)
    {
        TargetSP target_sp(target.GetSP());
        if (target_sp)
        {
            // No need to lock, the target list is thread safe
            result = m_opaque_sp->GetTargetList().DeleteTarget (target_sp);
            target_sp->Destroy();
            target.Clear();
            const bool mandatory = true;
            ModuleList::RemoveOrphanSharedModules(mandatory);
        }
    }

    Log *log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        log->Printf ("SBDebugger(%p)::DeleteTarget (SBTarget(%p)) => %i", m_opaque_sp.get(), target.m_opaque_sp.get(), result);
    }

    return result;
}
SBTarget
SBDebugger::GetTargetAtIndex (uint32_t idx)
{
    SBTarget sb_target;
    if (m_opaque_sp)
    {
        // No need to lock, the target list is thread safe
        sb_target.SetSP (m_opaque_sp->GetTargetList().GetTargetAtIndex (idx));
    }
    return sb_target;
}

uint32_t
SBDebugger::GetIndexOfTarget (lldb::SBTarget target)
{

    lldb::TargetSP target_sp = target.GetSP();
    if (!target_sp)
        return UINT32_MAX;

    if (!m_opaque_sp)
        return UINT32_MAX;

    return m_opaque_sp->GetTargetList().GetIndexOfTarget (target.GetSP());
}

SBTarget
SBDebugger::FindTargetWithProcessID (pid_t pid)
{
    SBTarget sb_target;
    if (m_opaque_sp)
    {
        // No need to lock, the target list is thread safe
        sb_target.SetSP (m_opaque_sp->GetTargetList().FindTargetWithProcessID (pid));
    }
    return sb_target;
}

SBTarget
SBDebugger::FindTargetWithFileAndArch (const char *filename, const char *arch_name)
{
    SBTarget sb_target;
    if (m_opaque_sp && filename && filename[0])
    {
        // No need to lock, the target list is thread safe
        ArchSpec arch (arch_name, m_opaque_sp->GetPlatformList().GetSelectedPlatform().get());
        TargetSP target_sp (m_opaque_sp->GetTargetList().FindTargetWithExecutableAndArchitecture (FileSpec(filename, false), arch_name ? &arch : NULL));
        sb_target.SetSP (target_sp);
    }
    return sb_target;
}

SBTarget
SBDebugger::FindTargetWithLLDBProcess (const ProcessSP &process_sp)
{
    SBTarget sb_target;
    if (m_opaque_sp)
    {
        // No need to lock, the target list is thread safe
        sb_target.SetSP (m_opaque_sp->GetTargetList().FindTargetWithProcess (process_sp.get()));
    }
    return sb_target;
}


uint32_t
SBDebugger::GetNumTargets ()
{
    if (m_opaque_sp)
    {
        // No need to lock, the target list is thread safe
        return m_opaque_sp->GetTargetList().GetNumTargets ();
    }
    return 0;
}

SBTarget
SBDebugger::GetSelectedTarget ()
{
    Log *log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBTarget sb_target;
    TargetSP target_sp;
    if (m_opaque_sp)
    {
        // No need to lock, the target list is thread safe
        target_sp = m_opaque_sp->GetTargetList().GetSelectedTarget ();
        sb_target.SetSP (target_sp);
    }

    if (log)
    {
        SBStream sstr;
        sb_target.GetDescription (sstr, eDescriptionLevelBrief);
        log->Printf ("SBDebugger(%p)::GetSelectedTarget () => SBTarget(%p): %s", m_opaque_sp.get(),
                     target_sp.get(), sstr.GetData());
    }

    return sb_target;
}

void
SBDebugger::SetSelectedTarget (SBTarget &sb_target)
{
    Log *log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    TargetSP target_sp (sb_target.GetSP());
    if (m_opaque_sp)
    {
        m_opaque_sp->GetTargetList().SetSelectedTarget (target_sp.get());
    }
    if (log)
    {
        SBStream sstr;
        sb_target.GetDescription (sstr, eDescriptionLevelBrief);
        log->Printf ("SBDebugger(%p)::SetSelectedTarget () => SBTarget(%p): %s", m_opaque_sp.get(),
                     target_sp.get(), sstr.GetData());
    }
}

void
SBDebugger::DispatchInput (void* baton, const void *data, size_t data_len)
{
    DispatchInput (data,data_len);
}

void
SBDebugger::DispatchInput (const void *data, size_t data_len)
{
    Log *log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    if (log)
        log->Printf ("SBDebugger(%p)::DispatchInput (data=\"%.*s\", size_t=%" PRIu64 ")",
                     m_opaque_sp.get(),
                     (int) data_len,
                     (const char *) data,
                     (uint64_t)data_len);

    if (m_opaque_sp)
        m_opaque_sp->DispatchInput ((const char *) data, data_len);
}

void
SBDebugger::DispatchInputInterrupt ()
{
    if (m_opaque_sp)
        m_opaque_sp->DispatchInputInterrupt ();
}

void
SBDebugger::DispatchInputEndOfFile ()
{
    if (m_opaque_sp)
        m_opaque_sp->DispatchInputEndOfFile ();
}
    
bool
SBDebugger::InputReaderIsTopReader (const lldb::SBInputReader &reader)
{
    Log *log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    if (log)
        log->Printf ("SBDebugger(%p)::InputReaderIsTopReader (SBInputReader(%p))", m_opaque_sp.get(), &reader);

    if (m_opaque_sp && reader.IsValid())
    {
        InputReaderSP reader_sp (*reader);
        return m_opaque_sp->InputReaderIsTopReader (reader_sp);
    }

    return false;
}


void
SBDebugger::PushInputReader (SBInputReader &reader)
{
    Log *log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    if (log)
        log->Printf ("SBDebugger(%p)::PushInputReader (SBInputReader(%p))", m_opaque_sp.get(), &reader);

    if (m_opaque_sp && reader.IsValid())
    {
        TargetSP target_sp (m_opaque_sp->GetSelectedTarget());
        Mutex::Locker api_locker;
        if (target_sp)
            api_locker.Lock(target_sp->GetAPIMutex());
        InputReaderSP reader_sp(*reader);
        m_opaque_sp->PushInputReader (reader_sp);
    }
}

void
SBDebugger::NotifyTopInputReader (InputReaderAction notification)
{
    Log *log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    if (log)
        log->Printf ("SBDebugger(%p)::NotifyTopInputReader (%d)", m_opaque_sp.get(), notification);
        
    if (m_opaque_sp)
        m_opaque_sp->NotifyTopInputReader (notification);
}

void
SBDebugger::reset (const DebuggerSP &debugger_sp)
{
    m_opaque_sp = debugger_sp;
}

Debugger *
SBDebugger::get () const
{
    return m_opaque_sp.get();
}

Debugger &
SBDebugger::ref () const
{
    assert (m_opaque_sp.get());
    return *m_opaque_sp;
}

const lldb::DebuggerSP &
SBDebugger::get_sp () const
{
    return m_opaque_sp;
}

SBDebugger
SBDebugger::FindDebuggerWithID (int id)
{
    // No need to lock, the debugger list is thread safe
    SBDebugger sb_debugger;
    DebuggerSP debugger_sp = Debugger::FindDebuggerWithID (id);
    if (debugger_sp)
        sb_debugger.reset (debugger_sp);
    return sb_debugger;
}

const char *
SBDebugger::GetInstanceName()
{
    if (m_opaque_sp)
        return m_opaque_sp->GetInstanceName().AsCString();
    else
        return NULL;
}

SBError
SBDebugger::SetInternalVariable (const char *var_name, const char *value, const char *debugger_instance_name)
{
    SBError sb_error;
    DebuggerSP debugger_sp(Debugger::FindDebuggerWithInstanceName (ConstString(debugger_instance_name)));
    Error error;
    if (debugger_sp)
    {
        ExecutionContext exe_ctx (debugger_sp->GetCommandInterpreter().GetExecutionContext());
        error = debugger_sp->SetPropertyValue (&exe_ctx,
                                               eVarSetOperationAssign,
                                               var_name,
                                               value);
    }
    else
    {
        error.SetErrorStringWithFormat ("invalid debugger instance name '%s'", debugger_instance_name);
    }
    if (error.Fail())
        sb_error.SetError(error);
    return sb_error;
}

SBStringList
SBDebugger::GetInternalVariableValue (const char *var_name, const char *debugger_instance_name)
{
    SBStringList ret_value;
    DebuggerSP debugger_sp(Debugger::FindDebuggerWithInstanceName (ConstString(debugger_instance_name)));
    Error error;
    if (debugger_sp)
    {
        ExecutionContext exe_ctx (debugger_sp->GetCommandInterpreter().GetExecutionContext());
        lldb::OptionValueSP value_sp (debugger_sp->GetPropertyValue (&exe_ctx,
                                                                     var_name,
                                                                     false,
                                                                     error));
        if (value_sp)
        {
            StreamString value_strm;
            value_sp->DumpValue (&exe_ctx, value_strm, OptionValue::eDumpOptionValue);
            const std::string &value_str = value_strm.GetString();
            if (!value_str.empty())
            {
                StringList string_list;
                string_list.SplitIntoLines(value_str.c_str(), value_str.size());
                return SBStringList(&string_list);
            }
        }
    }
    return SBStringList();
}

uint32_t
SBDebugger::GetTerminalWidth () const
{
    if (m_opaque_sp)
        return m_opaque_sp->GetTerminalWidth ();
    return 0;
}

void
SBDebugger::SetTerminalWidth (uint32_t term_width)
{
    if (m_opaque_sp)
        m_opaque_sp->SetTerminalWidth (term_width);
}

const char *
SBDebugger::GetPrompt() const
{
    Log *log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    
    if (log)
        log->Printf ("SBDebugger(%p)::GetPrompt () => \"%s\"", m_opaque_sp.get(), 
                     (m_opaque_sp ? m_opaque_sp->GetPrompt() : ""));

    if (m_opaque_sp)
        return m_opaque_sp->GetPrompt ();
    return 0;
}

void
SBDebugger::SetPrompt (const char *prompt)
{
    if (m_opaque_sp)
        m_opaque_sp->SetPrompt (prompt);
}

    
ScriptLanguage 
SBDebugger::GetScriptLanguage() const
{
    if (m_opaque_sp)
        return m_opaque_sp->GetScriptLanguage ();
    return eScriptLanguageNone;
}

void
SBDebugger::SetScriptLanguage (ScriptLanguage script_lang)
{
    if (m_opaque_sp)
    {
        m_opaque_sp->SetScriptLanguage (script_lang);
    }
}

bool
SBDebugger::SetUseExternalEditor (bool value)
{
    if (m_opaque_sp)
        return m_opaque_sp->SetUseExternalEditor (value);
    return false;
}

bool
SBDebugger::GetUseExternalEditor ()
{
    if (m_opaque_sp)
        return m_opaque_sp->GetUseExternalEditor ();
    return false;
}

bool
SBDebugger::GetDescription (SBStream &description)
{
    Stream &strm = description.ref();

    if (m_opaque_sp)
    {
        const char *name = m_opaque_sp->GetInstanceName().AsCString();
        user_id_t id = m_opaque_sp->GetID();
        strm.Printf ("Debugger (instance: \"%s\", id: %" PRIu64 ")", name, id);
    }
    else
        strm.PutCString ("No value");
    
    return true;
}

user_id_t
SBDebugger::GetID()
{
    if (m_opaque_sp)
        return m_opaque_sp->GetID();
    return LLDB_INVALID_UID;
}


SBError
SBDebugger::SetCurrentPlatform (const char *platform_name)
{
    SBError sb_error;
    if (m_opaque_sp)
    {
        PlatformSP platform_sp (Platform::Create (platform_name, sb_error.ref()));
        
        if (platform_sp)
        {
            bool make_selected = true;
            m_opaque_sp->GetPlatformList().Append (platform_sp, make_selected);
        }
    }
    return sb_error;
}

bool
SBDebugger::SetCurrentPlatformSDKRoot (const char *sysroot)
{
    if (m_opaque_sp)
    {
        PlatformSP platform_sp (m_opaque_sp->GetPlatformList().GetSelectedPlatform());
        
        if (platform_sp)
        {
            platform_sp->SetSDKRootDirectory (ConstString (sysroot));
            return true;
        }
    }
    return false;
}

bool
SBDebugger::GetCloseInputOnEOF () const
{
    if (m_opaque_sp)
        return m_opaque_sp->GetCloseInputOnEOF ();
    return false;
}

void
SBDebugger::SetCloseInputOnEOF (bool b)
{
    if (m_opaque_sp)
        m_opaque_sp->SetCloseInputOnEOF (b);
}

SBTypeCategory
SBDebugger::GetCategory (const char* category_name)
{
    if (!category_name || *category_name == 0)
        return SBTypeCategory();
    
    TypeCategoryImplSP category_sp;
    
    if (DataVisualization::Categories::GetCategory(ConstString(category_name), category_sp, false))
        return SBTypeCategory(category_sp);
    else
        return SBTypeCategory();
}

SBTypeCategory
SBDebugger::CreateCategory (const char* category_name)
{
    if (!category_name || *category_name == 0)
        return SBTypeCategory();
    
    TypeCategoryImplSP category_sp;
    
    if (DataVisualization::Categories::GetCategory(ConstString(category_name), category_sp, true))
        return SBTypeCategory(category_sp);
    else
        return SBTypeCategory();
}

bool
SBDebugger::DeleteCategory (const char* category_name)
{
    if (!category_name || *category_name == 0)
        return false;
    
    return DataVisualization::Categories::Delete(ConstString(category_name));
}

uint32_t
SBDebugger::GetNumCategories()
{
    return DataVisualization::Categories::GetCount();
}

SBTypeCategory
SBDebugger::GetCategoryAtIndex (uint32_t index)
{
    return SBTypeCategory(DataVisualization::Categories::GetCategoryAtIndex(index));
}

SBTypeCategory
SBDebugger::GetDefaultCategory()
{
    return GetCategory("default");
}

SBTypeFormat
SBDebugger::GetFormatForType (SBTypeNameSpecifier type_name)
{
    SBTypeCategory default_category_sb = GetDefaultCategory();
    if (default_category_sb.GetEnabled())
        return default_category_sb.GetFormatForType(type_name);
    return SBTypeFormat();
}

#ifndef LLDB_DISABLE_PYTHON
SBTypeSummary
SBDebugger::GetSummaryForType (SBTypeNameSpecifier type_name)
{
    if (type_name.IsValid() == false)
        return SBTypeSummary();
    return SBTypeSummary(DataVisualization::GetSummaryForType(type_name.GetSP()));
}
#endif // LLDB_DISABLE_PYTHON

SBTypeFilter
SBDebugger::GetFilterForType (SBTypeNameSpecifier type_name)
{
    if (type_name.IsValid() == false)
        return SBTypeFilter();
    return SBTypeFilter(DataVisualization::GetFilterForType(type_name.GetSP()));
}

#ifndef LLDB_DISABLE_PYTHON
SBTypeSynthetic
SBDebugger::GetSyntheticForType (SBTypeNameSpecifier type_name)
{
    if (type_name.IsValid() == false)
        return SBTypeSynthetic();
    return SBTypeSynthetic(DataVisualization::GetSyntheticForType(type_name.GetSP()));
}
#endif // LLDB_DISABLE_PYTHON

bool
SBDebugger::EnableLog (const char *channel, const char **categories)
{
    if (m_opaque_sp)
    {
        uint32_t log_options = LLDB_LOG_OPTION_PREPEND_TIMESTAMP | LLDB_LOG_OPTION_PREPEND_THREAD_NAME;
        StreamString errors;
        return m_opaque_sp->EnableLog (channel, categories, NULL, log_options, errors);
    
    }
    else
        return false;
}

void
SBDebugger::SetLoggingCallback (lldb::LogOutputCallback log_callback, void *baton)
{
    if (m_opaque_sp)
    {
        return m_opaque_sp->SetLoggingCallback (log_callback, baton);
    }
}


