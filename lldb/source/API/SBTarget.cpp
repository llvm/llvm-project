//===-- SBTarget.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBTarget.h"

#include "lldb/lldb-public.h"

#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBBreakpoint.h"
#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBListener.h"
#include "lldb/API/SBModule.h"
#include "lldb/API/SBSourceManager.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBSymbolContextList.h"
#include "lldb/Breakpoint/BreakpointID.h"
#include "lldb/Breakpoint/BreakpointIDList.h"
#include "lldb/Breakpoint/BreakpointList.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/AddressResolver.h"
#include "lldb/Core/AddressResolverName.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Disassembler.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/SearchFilter.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/STLUtils.h"
#include "lldb/Core/ValueObjectList.h"
#include "lldb/Core/ValueObjectVariable.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/Host.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/LanguageRuntime.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/TargetList.h"

#include "lldb/Interpreter/CommandReturnObject.h"
#include "../source/Commands/CommandObjectBreakpoint.h"


using namespace lldb;
using namespace lldb_private;

#define DEFAULT_DISASM_BYTE_SIZE 32

SBLaunchInfo::SBLaunchInfo (const char **argv) : 
    m_opaque_sp(new ProcessLaunchInfo())
{
    m_opaque_sp->GetFlags().Reset (eLaunchFlagDebug | eLaunchFlagDisableASLR);
    if (argv && argv[0])
        m_opaque_sp->GetArguments().SetArguments(argv);
}

SBLaunchInfo::~SBLaunchInfo()
{
}

lldb_private::ProcessLaunchInfo &
SBLaunchInfo::ref ()
{
    return *m_opaque_sp;
}


uint32_t
SBLaunchInfo::GetUserID()
{
    return m_opaque_sp->GetUserID();
}

uint32_t
SBLaunchInfo::GetGroupID()
{
    return m_opaque_sp->GetGroupID();
}

bool
SBLaunchInfo::UserIDIsValid ()
{
    return m_opaque_sp->UserIDIsValid();
}

bool
SBLaunchInfo::GroupIDIsValid ()
{
    return m_opaque_sp->GroupIDIsValid();
}

void
SBLaunchInfo::SetUserID (uint32_t uid)
{
    m_opaque_sp->SetUserID (uid);
}

void
SBLaunchInfo::SetGroupID (uint32_t gid)
{
    m_opaque_sp->SetGroupID (gid);
}

uint32_t
SBLaunchInfo::GetNumArguments ()
{
    return m_opaque_sp->GetArguments().GetArgumentCount();
}

const char *
SBLaunchInfo::GetArgumentAtIndex (uint32_t idx)
{
    return m_opaque_sp->GetArguments().GetArgumentAtIndex(idx);
}

void
SBLaunchInfo::SetArguments (const char **argv, bool append)
{
    if (append)
    {
        if (argv)
            m_opaque_sp->GetArguments().AppendArguments(argv);
    }
    else
    {
        if (argv)
            m_opaque_sp->GetArguments().SetArguments(argv);
        else
            m_opaque_sp->GetArguments().Clear();
    }
}

uint32_t
SBLaunchInfo::GetNumEnvironmentEntries ()
{
    return m_opaque_sp->GetEnvironmentEntries().GetArgumentCount();
}

const char *
SBLaunchInfo::GetEnvironmentEntryAtIndex (uint32_t idx)
{
    return m_opaque_sp->GetEnvironmentEntries().GetArgumentAtIndex(idx);
}

void
SBLaunchInfo::SetEnvironmentEntries (const char **envp, bool append)
{
    if (append)
    {
        if (envp)
            m_opaque_sp->GetEnvironmentEntries().AppendArguments(envp);
    }
    else
    {
        if (envp)
            m_opaque_sp->GetEnvironmentEntries().SetArguments(envp);
        else
            m_opaque_sp->GetEnvironmentEntries().Clear();
    }
}

void
SBLaunchInfo::Clear ()
{
    m_opaque_sp->Clear();
}

const char *
SBLaunchInfo::GetWorkingDirectory () const
{
    return m_opaque_sp->GetWorkingDirectory();
}

void
SBLaunchInfo::SetWorkingDirectory (const char *working_dir)
{
    m_opaque_sp->SetWorkingDirectory(working_dir);
}

uint32_t
SBLaunchInfo::GetLaunchFlags ()
{
    return m_opaque_sp->GetFlags().Get();
}

void
SBLaunchInfo::SetLaunchFlags (uint32_t flags)
{
    m_opaque_sp->GetFlags().Reset(flags);
}

const char *
SBLaunchInfo::GetProcessPluginName ()
{
    return m_opaque_sp->GetProcessPluginName();
}

void
SBLaunchInfo::SetProcessPluginName (const char *plugin_name)
{
    return m_opaque_sp->SetProcessPluginName (plugin_name);
}

const char *
SBLaunchInfo::GetShell ()
{
    return m_opaque_sp->GetShell();
}

void
SBLaunchInfo::SetShell (const char * path)
{
    m_opaque_sp->SetShell (path);
}

uint32_t
SBLaunchInfo::GetResumeCount ()
{
    return m_opaque_sp->GetResumeCount();
}

void
SBLaunchInfo::SetResumeCount (uint32_t c)
{
    m_opaque_sp->SetResumeCount (c);
}

bool
SBLaunchInfo::AddCloseFileAction (int fd)
{
    return m_opaque_sp->AppendCloseFileAction(fd);
}

bool
SBLaunchInfo::AddDuplicateFileAction (int fd, int dup_fd)
{
    return m_opaque_sp->AppendDuplicateFileAction(fd, dup_fd);
}

bool
SBLaunchInfo::AddOpenFileAction (int fd, const char *path, bool read, bool write)
{
    return m_opaque_sp->AppendOpenFileAction(fd, path, read, write);
}

bool
SBLaunchInfo::AddSuppressFileAction (int fd, bool read, bool write)
{
    return m_opaque_sp->AppendSuppressFileAction(fd, read, write);
}


SBAttachInfo::SBAttachInfo () :
    m_opaque_sp (new ProcessAttachInfo())
{
}

SBAttachInfo::SBAttachInfo (lldb::pid_t pid) :
    m_opaque_sp (new ProcessAttachInfo())
{
    m_opaque_sp->SetProcessID (pid);
}

SBAttachInfo::SBAttachInfo (const char *path, bool wait_for) :
    m_opaque_sp (new ProcessAttachInfo())
{
    if (path && path[0])
        m_opaque_sp->GetExecutableFile().SetFile(path, false);
    m_opaque_sp->SetWaitForLaunch (wait_for);
}

SBAttachInfo::SBAttachInfo (const SBAttachInfo &rhs) :
    m_opaque_sp (new ProcessAttachInfo())
{
    *m_opaque_sp = *rhs.m_opaque_sp;
}

SBAttachInfo::~SBAttachInfo()
{
}

lldb_private::ProcessAttachInfo &
SBAttachInfo::ref ()
{
    return *m_opaque_sp;
}

SBAttachInfo &
SBAttachInfo::operator = (const SBAttachInfo &rhs)
{
    if (this != &rhs)
        *m_opaque_sp = *rhs.m_opaque_sp;
    return *this;
}

lldb::pid_t
SBAttachInfo::GetProcessID ()
{
    return m_opaque_sp->GetProcessID();
}

void
SBAttachInfo::SetProcessID (lldb::pid_t pid)
{
    m_opaque_sp->SetProcessID (pid);
}


uint32_t
SBAttachInfo::GetResumeCount ()
{
    return m_opaque_sp->GetResumeCount();
}

void
SBAttachInfo::SetResumeCount (uint32_t c)
{
    m_opaque_sp->SetResumeCount (c);
}

const char *
SBAttachInfo::GetProcessPluginName ()
{
    return m_opaque_sp->GetProcessPluginName();
}

void
SBAttachInfo::SetProcessPluginName (const char *plugin_name)
{
    return m_opaque_sp->SetProcessPluginName (plugin_name);
}

void
SBAttachInfo::SetExecutable (const char *path)
{
    if (path && path[0])
        m_opaque_sp->GetExecutableFile().SetFile(path, false);
    else
        m_opaque_sp->GetExecutableFile().Clear();
}

void
SBAttachInfo::SetExecutable (SBFileSpec exe_file)
{
    if (exe_file.IsValid())
        m_opaque_sp->GetExecutableFile() = exe_file.ref();
    else
        m_opaque_sp->GetExecutableFile().Clear();
}

bool
SBAttachInfo::GetWaitForLaunch ()
{
    return m_opaque_sp->GetWaitForLaunch();
}

void
SBAttachInfo::SetWaitForLaunch (bool b)
{
    m_opaque_sp->SetWaitForLaunch (b);
}

bool
SBAttachInfo::GetIgnoreExisting ()
{
    return m_opaque_sp->GetIgnoreExisting();
}

void
SBAttachInfo::SetIgnoreExisting (bool b)
{
    m_opaque_sp->SetIgnoreExisting (b);
}

uint32_t
SBAttachInfo::GetUserID()
{
    return m_opaque_sp->GetUserID();
}

uint32_t
SBAttachInfo::GetGroupID()
{
    return m_opaque_sp->GetGroupID();
}

bool
SBAttachInfo::UserIDIsValid ()
{
    return m_opaque_sp->UserIDIsValid();
}

bool
SBAttachInfo::GroupIDIsValid ()
{
    return m_opaque_sp->GroupIDIsValid();
}

void
SBAttachInfo::SetUserID (uint32_t uid)
{
    m_opaque_sp->SetUserID (uid);
}

void
SBAttachInfo::SetGroupID (uint32_t gid)
{
    m_opaque_sp->SetGroupID (gid);
}

uint32_t
SBAttachInfo::GetEffectiveUserID()
{
    return m_opaque_sp->GetEffectiveUserID();
}

uint32_t
SBAttachInfo::GetEffectiveGroupID()
{
    return m_opaque_sp->GetEffectiveGroupID();
}

bool
SBAttachInfo::EffectiveUserIDIsValid ()
{
    return m_opaque_sp->EffectiveUserIDIsValid();
}

bool
SBAttachInfo::EffectiveGroupIDIsValid ()
{
    return m_opaque_sp->EffectiveGroupIDIsValid ();
}

void
SBAttachInfo::SetEffectiveUserID (uint32_t uid)
{
    m_opaque_sp->SetEffectiveUserID(uid);
}

void
SBAttachInfo::SetEffectiveGroupID (uint32_t gid)
{
    m_opaque_sp->SetEffectiveGroupID(gid);
}

lldb::pid_t
SBAttachInfo::GetParentProcessID ()
{
    return m_opaque_sp->GetParentProcessID();
}

void
SBAttachInfo::SetParentProcessID (lldb::pid_t pid)
{
    m_opaque_sp->SetParentProcessID (pid);
}

bool
SBAttachInfo::ParentProcessIDIsValid()
{
    return m_opaque_sp->ParentProcessIDIsValid();
}


//----------------------------------------------------------------------
// SBTarget constructor
//----------------------------------------------------------------------
SBTarget::SBTarget () :
    m_opaque_sp ()
{
}

SBTarget::SBTarget (const SBTarget& rhs) :
    m_opaque_sp (rhs.m_opaque_sp)
{
}

SBTarget::SBTarget(const TargetSP& target_sp) :
    m_opaque_sp (target_sp)
{
}

const SBTarget&
SBTarget::operator = (const SBTarget& rhs)
{
    if (this != &rhs)
        m_opaque_sp = rhs.m_opaque_sp;
    return *this;
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
SBTarget::~SBTarget()
{
}

const char *
SBTarget::GetBroadcasterClassName ()
{
    return Target::GetStaticBroadcasterClass().AsCString();
}

bool
SBTarget::IsValid () const
{
    return m_opaque_sp.get() != NULL && m_opaque_sp->IsValid();
}

SBProcess
SBTarget::GetProcess ()
{
    SBProcess sb_process;
    ProcessSP process_sp;
    TargetSP target_sp(GetSP());
    if (target_sp)
    {
        process_sp = target_sp->GetProcessSP();
        sb_process.SetSP (process_sp);
    }

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        log->Printf ("SBTarget(%p)::GetProcess () => SBProcess(%p)", 
                     target_sp.get(), process_sp.get());
    }

    return sb_process;
}

SBDebugger
SBTarget::GetDebugger () const
{
    SBDebugger debugger;
    TargetSP target_sp(GetSP());
    if (target_sp)
        debugger.reset (target_sp->GetDebugger().shared_from_this());
    return debugger;
}

SBProcess
SBTarget::LaunchSimple
(
    char const **argv,
    char const **envp,
    const char *working_directory
)
{
    char *stdin_path = NULL;
    char *stdout_path = NULL;
    char *stderr_path = NULL;
    uint32_t launch_flags = 0;
    bool stop_at_entry = false;
    SBError error;
    SBListener listener = GetDebugger().GetListener();
    return Launch (listener,
                   argv,
                   envp,
                   stdin_path,
                   stdout_path,
                   stderr_path,
                   working_directory,
                   launch_flags,
                   stop_at_entry,
                   error);
}

SBProcess
SBTarget::Launch 
(
    SBListener &listener, 
    char const **argv,
    char const **envp,
    const char *stdin_path,
    const char *stdout_path,
    const char *stderr_path,
    const char *working_directory,
    uint32_t launch_flags,   // See LaunchFlags
    bool stop_at_entry,
    lldb::SBError& error
)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBProcess sb_process;
    ProcessSP process_sp;
    TargetSP target_sp(GetSP());

    if (log)
    {
        log->Printf ("SBTarget(%p)::Launch (argv=%p, envp=%p, stdin=%s, stdout=%s, stderr=%s, working-dir=%s, launch_flags=0x%x, stop_at_entry=%i, &error (%p))...",
                     target_sp.get(), 
                     argv, 
                     envp, 
                     stdin_path ? stdin_path : "NULL", 
                     stdout_path ? stdout_path : "NULL", 
                     stderr_path ? stderr_path : "NULL", 
                     working_directory ? working_directory : "NULL",
                     launch_flags, 
                     stop_at_entry, 
                     error.get());
    }

    if (target_sp)
    {
        Mutex::Locker api_locker (target_sp->GetAPIMutex());

        if (getenv("LLDB_LAUNCH_FLAG_DISABLE_ASLR"))
            launch_flags |= eLaunchFlagDisableASLR;

        StateType state = eStateInvalid;
        process_sp = target_sp->GetProcessSP();
        if (process_sp)
        {
            state = process_sp->GetState();
            
            if (process_sp->IsAlive() && state != eStateConnected)
            {       
                if (state == eStateAttaching)
                    error.SetErrorString ("process attach is in progress");
                else
                    error.SetErrorString ("a process is already being debugged");
                return sb_process;
            }            
        }
        
        if (state == eStateConnected)
        {
            // If we are already connected, then we have already specified the
            // listener, so if a valid listener is supplied, we need to error out
            // to let the client know.
            if (listener.IsValid())
            {
                error.SetErrorString ("process is connected and already has a listener, pass empty listener");
                return sb_process;
            }
        }
        else
        {
            if (listener.IsValid())
                process_sp = target_sp->CreateProcess (listener.ref(), NULL, NULL);
            else
                process_sp = target_sp->CreateProcess (target_sp->GetDebugger().GetListener(), NULL, NULL);
        }

        if (process_sp)
        {
            sb_process.SetSP (process_sp);
            if (getenv("LLDB_LAUNCH_FLAG_DISABLE_STDIO"))
                launch_flags |= eLaunchFlagDisableSTDIO;

            ProcessLaunchInfo launch_info (stdin_path, stdout_path, stderr_path, working_directory, launch_flags);
            
            Module *exe_module = target_sp->GetExecutableModulePointer();
            if (exe_module)
                launch_info.SetExecutableFile(exe_module->GetPlatformFileSpec(), true);
            if (argv)
                launch_info.GetArguments().AppendArguments (argv);
            if (envp)
                launch_info.GetEnvironmentEntries ().SetArguments (envp);

            error.SetError (process_sp->Launch (launch_info));
            if (error.Success())
            {
                // We we are stopping at the entry point, we can return now!
                if (stop_at_entry)
                    return sb_process;
                
                // Make sure we are stopped at the entry
                StateType state = process_sp->WaitForProcessToStop (NULL);
                if (state == eStateStopped)
                {
                    // resume the process to skip the entry point
                    error.SetError (process_sp->Resume());
                    if (error.Success())
                    {
                        // If we are doing synchronous mode, then wait for the
                        // process to stop yet again!
                        if (target_sp->GetDebugger().GetAsyncExecution () == false)
                            process_sp->WaitForProcessToStop (NULL);
                    }
                }
            }
        }
        else
        {
            error.SetErrorString ("unable to create lldb_private::Process");
        }
    }
    else
    {
        error.SetErrorString ("SBTarget is invalid");
    }

    log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);
    if (log)
    {
        log->Printf ("SBTarget(%p)::Launch (...) => SBProcess(%p)", 
                     target_sp.get(), process_sp.get());
    }

    return sb_process;
}

SBProcess
SBTarget::Launch (SBLaunchInfo &sb_launch_info, SBError& error)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    
    SBProcess sb_process;
    ProcessSP process_sp;
    TargetSP target_sp(GetSP());
    
    if (log)
    {
        log->Printf ("SBTarget(%p)::Launch (launch_info, error)...", target_sp.get());
    }
    
    if (target_sp)
    {
        Mutex::Locker api_locker (target_sp->GetAPIMutex());
        StateType state = eStateInvalid;
        process_sp = target_sp->GetProcessSP();
        if (process_sp)
        {
            state = process_sp->GetState();
            
            if (process_sp->IsAlive() && state != eStateConnected)
            {       
                if (state == eStateAttaching)
                    error.SetErrorString ("process attach is in progress");
                else
                    error.SetErrorString ("a process is already being debugged");
                return sb_process;
            }            
        }
        
        if (state != eStateConnected)
            process_sp = target_sp->CreateProcess (target_sp->GetDebugger().GetListener(), NULL, NULL);
        
        if (process_sp)
        {
            sb_process.SetSP (process_sp);
            lldb_private::ProcessLaunchInfo &launch_info = sb_launch_info.ref();

            Module *exe_module = target_sp->GetExecutableModulePointer();
            if (exe_module)
                launch_info.SetExecutableFile(exe_module->GetPlatformFileSpec(), true);

            const ArchSpec &arch_spec = target_sp->GetArchitecture();
            if (arch_spec.IsValid())
                launch_info.GetArchitecture () = arch_spec;
    
            error.SetError (process_sp->Launch (launch_info));
            const bool synchronous_execution = target_sp->GetDebugger().GetAsyncExecution () == false;
            if (error.Success())
            {
                if (launch_info.GetFlags().Test(eLaunchFlagStopAtEntry))
                {
                    // If we are doing synchronous mode, then wait for the initial
                    // stop to happen, else, return and let the caller watch for
                    // the stop
                    if (synchronous_execution)
                         process_sp->WaitForProcessToStop (NULL);
                    // We we are stopping at the entry point, we can return now!
                    return sb_process;
                }
                
                // Make sure we are stopped at the entry
                StateType state = process_sp->WaitForProcessToStop (NULL);
                if (state == eStateStopped)
                {
                    // resume the process to skip the entry point
                    error.SetError (process_sp->Resume());
                    if (error.Success())
                    {
                        // If we are doing synchronous mode, then wait for the
                        // process to stop yet again!
                        if (synchronous_execution)
                            process_sp->WaitForProcessToStop (NULL);
                    }
                }
            }
        }
        else
        {
            error.SetErrorString ("unable to create lldb_private::Process");
        }
    }
    else
    {
        error.SetErrorString ("SBTarget is invalid");
    }
    
    log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);
    if (log)
    {
        log->Printf ("SBTarget(%p)::Launch (...) => SBProcess(%p)", 
                     target_sp.get(), process_sp.get());
    }
    
    return sb_process;
}

lldb::SBProcess
SBTarget::Attach (SBAttachInfo &sb_attach_info, SBError& error)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    
    SBProcess sb_process;
    ProcessSP process_sp;
    TargetSP target_sp(GetSP());
    
    if (log)
    {
        log->Printf ("SBTarget(%p)::Attach (sb_attach_info, error)...", target_sp.get());
    }
    
    if (target_sp)
    {
        Mutex::Locker api_locker (target_sp->GetAPIMutex());
        
        StateType state = eStateInvalid;
        process_sp = target_sp->GetProcessSP();
        if (process_sp)
        {
            state = process_sp->GetState();
            
            if (process_sp->IsAlive() && state != eStateConnected)
            {       
                if (state == eStateAttaching)
                    error.SetErrorString ("process attach is in progress");
                else
                    error.SetErrorString ("a process is already being debugged");
                if (log)
                {
                    log->Printf ("SBTarget(%p)::Attach (...) => error %s",
                                 target_sp.get(), error.GetCString());
                }
                return sb_process;
            }            
        }
        
        if (state != eStateConnected)
            process_sp = target_sp->CreateProcess (target_sp->GetDebugger().GetListener(), NULL, NULL);

        if (process_sp)
        {
            ProcessAttachInfo &attach_info = sb_attach_info.ref();
            lldb::pid_t attach_pid = attach_info.GetProcessID();
            if (attach_pid != LLDB_INVALID_PROCESS_ID)
            {
                PlatformSP platform_sp = target_sp->GetPlatform();
                // See if we can pre-verify if a process exists or not
                if (platform_sp && platform_sp->IsConnected())
                {
                    ProcessInstanceInfo instance_info;
                    if (platform_sp->GetProcessInfo(attach_pid, instance_info))
                    {
                        attach_info.SetUserID(instance_info.GetEffectiveUserID());
                    }
                    else
                    {
                        error.ref().SetErrorStringWithFormat("no process found with process ID %llu", attach_pid);
                        if (log)
                        {
                            log->Printf ("SBTarget(%p)::Attach (...) => error %s",
                                         target_sp.get(), error.GetCString());
                        }
                        return sb_process;
                    }
                }
            }
            error.SetError (process_sp->Attach (attach_info));
            if (error.Success())
            {
                sb_process.SetSP (process_sp);
                // If we are doing synchronous mode, then wait for the
                // process to stop!
                if (target_sp->GetDebugger().GetAsyncExecution () == false)
                    process_sp->WaitForProcessToStop (NULL);
            }
        }
        else
        {
            error.SetErrorString ("unable to create lldb_private::Process");
        }
    }
    else
    {
        error.SetErrorString ("SBTarget is invalid");
    }
    
    if (log)
    {
        log->Printf ("SBTarget(%p)::Attach (...) => SBProcess(%p)",
                     target_sp.get(), process_sp.get());
    }
    
    return sb_process;
}


#if defined(__APPLE__)

lldb::SBProcess
SBTarget::AttachToProcessWithID (SBListener &listener,
                                ::pid_t pid,
                                 lldb::SBError& error)
{
    return AttachToProcessWithID (listener, (lldb::pid_t)pid, error);
}

#endif // #if defined(__APPLE__)

lldb::SBProcess
SBTarget::AttachToProcessWithID 
(
    SBListener &listener, 
    lldb::pid_t pid,// The process ID to attach to
    SBError& error  // An error explaining what went wrong if attach fails
)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBProcess sb_process;
    ProcessSP process_sp;
    TargetSP target_sp(GetSP());

    if (log)
    {
        log->Printf ("SBTarget(%p)::AttachToProcessWithID (listener, pid=%lld, error)...", target_sp.get(), pid);
    }
    
    if (target_sp)
    {
        Mutex::Locker api_locker (target_sp->GetAPIMutex());

        StateType state = eStateInvalid;
        process_sp = target_sp->GetProcessSP();
        if (process_sp)
        {
            state = process_sp->GetState();
            
            if (process_sp->IsAlive() && state != eStateConnected)
            {       
                if (state == eStateAttaching)
                    error.SetErrorString ("process attach is in progress");
                else
                    error.SetErrorString ("a process is already being debugged");
                return sb_process;
            }            
        }

        if (state == eStateConnected)
        {
            // If we are already connected, then we have already specified the
            // listener, so if a valid listener is supplied, we need to error out
            // to let the client know.
            if (listener.IsValid())
            {
                error.SetErrorString ("process is connected and already has a listener, pass empty listener");
                return sb_process;
            }
        }
        else
        {
            if (listener.IsValid())
                process_sp = target_sp->CreateProcess (listener.ref(), NULL, NULL);
            else
                process_sp = target_sp->CreateProcess (target_sp->GetDebugger().GetListener(), NULL, NULL);
        }
        if (process_sp)
        {
            sb_process.SetSP (process_sp);
            
            ProcessAttachInfo attach_info;
            attach_info.SetProcessID (pid);
            
            PlatformSP platform_sp = target_sp->GetPlatform();
            ProcessInstanceInfo instance_info;
            if (platform_sp->GetProcessInfo(pid, instance_info))
            {
                attach_info.SetUserID(instance_info.GetEffectiveUserID());
            }
            error.SetError (process_sp->Attach (attach_info));            
            if (error.Success())
            {
                // If we are doing synchronous mode, then wait for the
                // process to stop!
                if (target_sp->GetDebugger().GetAsyncExecution () == false)
                process_sp->WaitForProcessToStop (NULL);
            }
        }
        else
        {
            error.SetErrorString ("unable to create lldb_private::Process");
        }
    }
    else
    {
        error.SetErrorString ("SBTarget is invalid");
    }
    
    if (log)
    {
        log->Printf ("SBTarget(%p)::AttachToProcessWithID (...) => SBProcess(%p)",
                     target_sp.get(), process_sp.get());
    }
    return sb_process;
}

lldb::SBProcess
SBTarget::AttachToProcessWithName 
(
    SBListener &listener, 
    const char *name,   // basename of process to attach to
    bool wait_for,      // if true wait for a new instance of "name" to be launched
    SBError& error      // An error explaining what went wrong if attach fails
)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    
    SBProcess sb_process;
    ProcessSP process_sp;
    TargetSP target_sp(GetSP());
    
    if (log)
    {
        log->Printf ("SBTarget(%p)::AttachToProcessWithName (listener, name=%s, wait_for=%s, error)...", target_sp.get(), name, wait_for ? "true" : "false");
    }
    
    if (name && target_sp)
    {
        Mutex::Locker api_locker (target_sp->GetAPIMutex());

        StateType state = eStateInvalid;
        process_sp = target_sp->GetProcessSP();
        if (process_sp)
        {
            state = process_sp->GetState();
            
            if (process_sp->IsAlive() && state != eStateConnected)
            {       
                if (state == eStateAttaching)
                    error.SetErrorString ("process attach is in progress");
                else
                    error.SetErrorString ("a process is already being debugged");
                return sb_process;
            }            
        }
        
        if (state == eStateConnected)
        {
            // If we are already connected, then we have already specified the
            // listener, so if a valid listener is supplied, we need to error out
            // to let the client know.
            if (listener.IsValid())
            {
                error.SetErrorString ("process is connected and already has a listener, pass empty listener");
                return sb_process;
            }
        }
        else
        {
            if (listener.IsValid())
                process_sp = target_sp->CreateProcess (listener.ref(), NULL, NULL);
            else
                process_sp = target_sp->CreateProcess (target_sp->GetDebugger().GetListener(), NULL, NULL);
        }

        if (process_sp)
        {
            sb_process.SetSP (process_sp);
            ProcessAttachInfo attach_info;
            attach_info.GetExecutableFile().SetFile(name, false);
            attach_info.SetWaitForLaunch(wait_for);
            error.SetError (process_sp->Attach (attach_info));
            // If we are doing synchronous mode, then wait for the
            // process to stop!
            if (target_sp->GetDebugger().GetAsyncExecution () == false)
                process_sp->WaitForProcessToStop (NULL);
        }
        else
        {
            error.SetErrorString ("unable to create lldb_private::Process");
        }
    }
    else
    {
        error.SetErrorString ("SBTarget is invalid");
    }
    
    if (log)
    {
        log->Printf ("SBTarget(%p)::AttachToPorcessWithName (...) => SBProcess(%p)",
                     target_sp.get(), process_sp.get());
    }
    return sb_process;
}

lldb::SBProcess
SBTarget::ConnectRemote
(
    SBListener &listener,
    const char *url,
    const char *plugin_name,
    SBError& error
)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBProcess sb_process;
    ProcessSP process_sp;
    TargetSP target_sp(GetSP());
    
    if (log)
    {
        log->Printf ("SBTarget(%p)::ConnectRemote (listener, url=%s, plugin_name=%s, error)...", target_sp.get(), url, plugin_name);
    }
    
    if (target_sp)
    {
        Mutex::Locker api_locker (target_sp->GetAPIMutex());
        if (listener.IsValid())
            process_sp = target_sp->CreateProcess (listener.ref(), plugin_name, NULL);
        else
            process_sp = target_sp->CreateProcess (target_sp->GetDebugger().GetListener(), plugin_name, NULL);
        
        
        if (process_sp)
        {
            sb_process.SetSP (process_sp);
            error.SetError (process_sp->ConnectRemote (NULL, url));
        }
        else
        {
            error.SetErrorString ("unable to create lldb_private::Process");
        }
    }
    else
    {
        error.SetErrorString ("SBTarget is invalid");
    }
    
    if (log)
    {
        log->Printf ("SBTarget(%p)::ConnectRemote (...) => SBProcess(%p)",
                     target_sp.get(), process_sp.get());
    }
    return sb_process;
}

SBFileSpec
SBTarget::GetExecutable ()
{

    SBFileSpec exe_file_spec;
    TargetSP target_sp(GetSP());
    if (target_sp)
    {
        Module *exe_module = target_sp->GetExecutableModulePointer();
        if (exe_module)
            exe_file_spec.SetFileSpec (exe_module->GetFileSpec());
    }

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        log->Printf ("SBTarget(%p)::GetExecutable () => SBFileSpec(%p)", 
                     target_sp.get(), exe_file_spec.get());
    }

    return exe_file_spec;
}

bool
SBTarget::operator == (const SBTarget &rhs) const
{
    return m_opaque_sp.get() == rhs.m_opaque_sp.get();
}

bool
SBTarget::operator != (const SBTarget &rhs) const
{
    return m_opaque_sp.get() != rhs.m_opaque_sp.get();
}

lldb::TargetSP
SBTarget::GetSP () const
{
    return m_opaque_sp;
}

void
SBTarget::SetSP (const lldb::TargetSP& target_sp)
{
    m_opaque_sp = target_sp;
}

lldb::SBAddress
SBTarget::ResolveLoadAddress (lldb::addr_t vm_addr)
{
    lldb::SBAddress sb_addr;
    Address &addr = sb_addr.ref();
    TargetSP target_sp(GetSP());
    if (target_sp)
    {
        Mutex::Locker api_locker (target_sp->GetAPIMutex());
        if (target_sp->GetSectionLoadList().ResolveLoadAddress (vm_addr, addr))
            return sb_addr;
    }

    // We have a load address that isn't in a section, just return an address
    // with the offset filled in (the address) and the section set to NULL
    addr.SetRawAddress(vm_addr);
    return sb_addr;
}

SBSymbolContext
SBTarget::ResolveSymbolContextForAddress (const SBAddress& addr, uint32_t resolve_scope)
{
    SBSymbolContext sc;
    if (addr.IsValid())
    {
        TargetSP target_sp(GetSP());
        if (target_sp)
            target_sp->GetImages().ResolveSymbolContextForAddress (addr.ref(), resolve_scope, sc.ref());
    }
    return sc;
}


SBBreakpoint
SBTarget::BreakpointCreateByLocation (const char *file, uint32_t line)
{
    return SBBreakpoint(BreakpointCreateByLocation (SBFileSpec (file, false), line));
}

SBBreakpoint
SBTarget::BreakpointCreateByLocation (const SBFileSpec &sb_file_spec, uint32_t line)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBBreakpoint sb_bp;
    TargetSP target_sp(GetSP());
    if (target_sp && line != 0)
    {
        Mutex::Locker api_locker (target_sp->GetAPIMutex());
        
        const LazyBool check_inlines = eLazyBoolCalculate;
        const LazyBool skip_prologue = eLazyBoolCalculate;
        const bool internal = false;
        *sb_bp = target_sp->CreateBreakpoint (NULL, *sb_file_spec, line, check_inlines, skip_prologue, internal);
    }

    if (log)
    {
        SBStream sstr;
        sb_bp.GetDescription (sstr);
        char path[PATH_MAX];
        sb_file_spec->GetPath (path, sizeof(path));
        log->Printf ("SBTarget(%p)::BreakpointCreateByLocation ( %s:%u ) => SBBreakpoint(%p): %s", 
                     target_sp.get(), 
                     path,
                     line, 
                     sb_bp.get(),
                     sstr.GetData());
    }

    return sb_bp;
}

SBBreakpoint
SBTarget::BreakpointCreateByName (const char *symbol_name, const char *module_name)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBBreakpoint sb_bp;
    TargetSP target_sp(GetSP());
    if (target_sp.get())
    {
        Mutex::Locker api_locker (target_sp->GetAPIMutex());
        
        const bool internal = false;
        const LazyBool skip_prologue = eLazyBoolCalculate;
        if (module_name && module_name[0])
        {
            FileSpecList module_spec_list;
            module_spec_list.Append (FileSpec (module_name, false));
            *sb_bp = target_sp->CreateBreakpoint (&module_spec_list, NULL, symbol_name, eFunctionNameTypeAuto, skip_prologue, internal);
        }
        else
        {
            *sb_bp = target_sp->CreateBreakpoint (NULL, NULL, symbol_name, eFunctionNameTypeAuto, skip_prologue, internal);
        }
    }
    
    if (log)
    {
        log->Printf ("SBTarget(%p)::BreakpointCreateByName (symbol=\"%s\", module=\"%s\") => SBBreakpoint(%p)", 
                     target_sp.get(), symbol_name, module_name, sb_bp.get());
    }

    return sb_bp;
}

lldb::SBBreakpoint
SBTarget::BreakpointCreateByName (const char *symbol_name, 
                            const SBFileSpecList &module_list, 
                            const SBFileSpecList &comp_unit_list)
{
    uint32_t name_type_mask = eFunctionNameTypeAuto;
    return BreakpointCreateByName (symbol_name, name_type_mask, module_list, comp_unit_list);
}

lldb::SBBreakpoint
SBTarget::BreakpointCreateByName (const char *symbol_name,
                            uint32_t name_type_mask,
                            const SBFileSpecList &module_list, 
                            const SBFileSpecList &comp_unit_list)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBBreakpoint sb_bp;
    TargetSP target_sp(GetSP());
    if (target_sp && symbol_name && symbol_name[0])
    {
        const bool internal = false;
        const LazyBool skip_prologue = eLazyBoolCalculate;
        Mutex::Locker api_locker (target_sp->GetAPIMutex());
        *sb_bp = target_sp->CreateBreakpoint (module_list.get(), 
                                                comp_unit_list.get(), 
                                                symbol_name, 
                                                name_type_mask, 
                                                skip_prologue,
                                                internal);
    }
    
    if (log)
    {
        log->Printf ("SBTarget(%p)::BreakpointCreateByName (symbol=\"%s\", name_type: %d) => SBBreakpoint(%p)", 
                     target_sp.get(), symbol_name, name_type_mask, sb_bp.get());
    }

    return sb_bp;
}

lldb::SBBreakpoint
SBTarget::BreakpointCreateByNames (const char *symbol_names[],
                                   uint32_t num_names,
                                   uint32_t name_type_mask,
                                   const SBFileSpecList &module_list,
                                   const SBFileSpecList &comp_unit_list)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBBreakpoint sb_bp;
    TargetSP target_sp(GetSP());
    if (target_sp && num_names > 0)
    {
        Mutex::Locker api_locker (target_sp->GetAPIMutex());
        const bool internal = false;
        const LazyBool skip_prologue = eLazyBoolCalculate;
        *sb_bp = target_sp->CreateBreakpoint (module_list.get(), 
                                                comp_unit_list.get(), 
                                                symbol_names,
                                                num_names,
                                                name_type_mask, 
                                                skip_prologue,
                                                internal);
    }
    
    if (log)
    {
        log->Printf ("SBTarget(%p)::BreakpointCreateByName (symbols={", target_sp.get());
        for (uint32_t i = 0 ; i < num_names; i++)
        {
            char sep;
            if (i < num_names - 1)
                sep = ',';
            else
                sep = '}';
            if (symbol_names[i] != NULL)
                log->Printf ("\"%s\"%c ", symbol_names[i], sep);
            else
                log->Printf ("\"<NULL>\"%c ", sep);
            
        }
        log->Printf ("name_type: %d) => SBBreakpoint(%p)", name_type_mask, sb_bp.get());
    }

    return sb_bp;
}

SBBreakpoint
SBTarget::BreakpointCreateByRegex (const char *symbol_name_regex, const char *module_name)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBBreakpoint sb_bp;
    TargetSP target_sp(GetSP());
    if (target_sp && symbol_name_regex && symbol_name_regex[0])
    {
        Mutex::Locker api_locker (target_sp->GetAPIMutex());
        RegularExpression regexp(symbol_name_regex);
        const bool internal = false;
        const LazyBool skip_prologue = eLazyBoolCalculate;
        
        if (module_name && module_name[0])
        {
            FileSpecList module_spec_list;
            module_spec_list.Append (FileSpec (module_name, false));
            
            *sb_bp = target_sp->CreateFuncRegexBreakpoint (&module_spec_list, NULL, regexp, skip_prologue, internal);
        }
        else
        {
            *sb_bp = target_sp->CreateFuncRegexBreakpoint (NULL, NULL, regexp, skip_prologue, internal);
        }
    }

    if (log)
    {
        log->Printf ("SBTarget(%p)::BreakpointCreateByRegex (symbol_regex=\"%s\", module_name=\"%s\") => SBBreakpoint(%p)", 
                     target_sp.get(), symbol_name_regex, module_name, sb_bp.get());
    }

    return sb_bp;
}

lldb::SBBreakpoint
SBTarget::BreakpointCreateByRegex (const char *symbol_name_regex, 
                         const SBFileSpecList &module_list, 
                         const SBFileSpecList &comp_unit_list)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBBreakpoint sb_bp;
    TargetSP target_sp(GetSP());
    if (target_sp && symbol_name_regex && symbol_name_regex[0])
    {
        Mutex::Locker api_locker (target_sp->GetAPIMutex());
        RegularExpression regexp(symbol_name_regex);
        const bool internal = false;
        const LazyBool skip_prologue = eLazyBoolCalculate;
        
        *sb_bp = target_sp->CreateFuncRegexBreakpoint (module_list.get(), comp_unit_list.get(), regexp, skip_prologue, internal);
    }

    if (log)
    {
        log->Printf ("SBTarget(%p)::BreakpointCreateByRegex (symbol_regex=\"%s\") => SBBreakpoint(%p)", 
                     target_sp.get(), symbol_name_regex, sb_bp.get());
    }

    return sb_bp;
}

SBBreakpoint
SBTarget::BreakpointCreateByAddress (addr_t address)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBBreakpoint sb_bp;
    TargetSP target_sp(GetSP());
    if (target_sp)
    {
        Mutex::Locker api_locker (target_sp->GetAPIMutex());
        *sb_bp = target_sp->CreateBreakpoint (address, false);
    }
    
    if (log)
    {
        log->Printf ("SBTarget(%p)::BreakpointCreateByAddress (address=%llu) => SBBreakpoint(%p)", target_sp.get(), (uint64_t) address, sb_bp.get());
    }

    return sb_bp;
}

lldb::SBBreakpoint
SBTarget::BreakpointCreateBySourceRegex (const char *source_regex, const lldb::SBFileSpec &source_file, const char *module_name)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBBreakpoint sb_bp;
    TargetSP target_sp(GetSP());
    if (target_sp && source_regex && source_regex[0])
    {
        Mutex::Locker api_locker (target_sp->GetAPIMutex());
        RegularExpression regexp(source_regex);
        FileSpecList source_file_spec_list;
        source_file_spec_list.Append (source_file.ref());
        
        if (module_name && module_name[0])
        {
            FileSpecList module_spec_list;
            module_spec_list.Append (FileSpec (module_name, false));
            
            *sb_bp = target_sp->CreateSourceRegexBreakpoint (&module_spec_list, &source_file_spec_list, regexp, false);
        }
        else
        {
            *sb_bp = target_sp->CreateSourceRegexBreakpoint (NULL, &source_file_spec_list, regexp, false);
        }
    }

    if (log)
    {
        char path[PATH_MAX];
        source_file->GetPath (path, sizeof(path));
        log->Printf ("SBTarget(%p)::BreakpointCreateByRegex (source_regex=\"%s\", file=\"%s\", module_name=\"%s\") => SBBreakpoint(%p)", 
                     target_sp.get(), source_regex, path, module_name, sb_bp.get());
    }

    return sb_bp;
}

lldb::SBBreakpoint
SBTarget::BreakpointCreateBySourceRegex (const char *source_regex, 
                               const SBFileSpecList &module_list, 
                               const lldb::SBFileSpecList &source_file_list)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBBreakpoint sb_bp;
    TargetSP target_sp(GetSP());
    if (target_sp && source_regex && source_regex[0])
    {
        Mutex::Locker api_locker (target_sp->GetAPIMutex());
        RegularExpression regexp(source_regex);
        *sb_bp = target_sp->CreateSourceRegexBreakpoint (module_list.get(), source_file_list.get(), regexp, false);
    }

    if (log)
    {
        log->Printf ("SBTarget(%p)::BreakpointCreateByRegex (source_regex=\"%s\") => SBBreakpoint(%p)", 
                     target_sp.get(), source_regex, sb_bp.get());
    }

    return sb_bp;
}

lldb::SBBreakpoint
SBTarget::BreakpointCreateForException  (lldb::LanguageType language,
                               bool catch_bp,
                               bool throw_bp)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBBreakpoint sb_bp;
    TargetSP target_sp(GetSP());
    if (target_sp)
    {
        Mutex::Locker api_locker (target_sp->GetAPIMutex());
        *sb_bp = target_sp->CreateExceptionBreakpoint (language, catch_bp, throw_bp);
    }

    if (log)
    {
        log->Printf ("SBTarget(%p)::BreakpointCreateByRegex (Language: %s, catch: %s throw: %s) => SBBreakpoint(%p)", 
                     target_sp.get(),
                     LanguageRuntime::GetNameForLanguageType(language),
                     catch_bp ? "on" : "off",
                     throw_bp ? "on" : "off",
                     sb_bp.get());
    }

    return sb_bp;
}

uint32_t
SBTarget::GetNumBreakpoints () const
{
    TargetSP target_sp(GetSP());
    if (target_sp)
    {
        // The breakpoint list is thread safe, no need to lock
        return target_sp->GetBreakpointList().GetSize();
    }
    return 0;
}

SBBreakpoint
SBTarget::GetBreakpointAtIndex (uint32_t idx) const
{
    SBBreakpoint sb_breakpoint;
    TargetSP target_sp(GetSP());
    if (target_sp)
    {
        // The breakpoint list is thread safe, no need to lock
        *sb_breakpoint = target_sp->GetBreakpointList().GetBreakpointAtIndex(idx);
    }
    return sb_breakpoint;
}

bool
SBTarget::BreakpointDelete (break_id_t bp_id)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    bool result = false;
    TargetSP target_sp(GetSP());
    if (target_sp)
    {
        Mutex::Locker api_locker (target_sp->GetAPIMutex());
        result = target_sp->RemoveBreakpointByID (bp_id);
    }

    if (log)
    {
        log->Printf ("SBTarget(%p)::BreakpointDelete (bp_id=%d) => %i", target_sp.get(), (uint32_t) bp_id, result);
    }

    return result;
}

SBBreakpoint
SBTarget::FindBreakpointByID (break_id_t bp_id)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBBreakpoint sb_breakpoint;
    TargetSP target_sp(GetSP());
    if (target_sp && bp_id != LLDB_INVALID_BREAK_ID)
    {
        Mutex::Locker api_locker (target_sp->GetAPIMutex());
        *sb_breakpoint = target_sp->GetBreakpointByID (bp_id);
    }

    if (log)
    {
        log->Printf ("SBTarget(%p)::FindBreakpointByID (bp_id=%d) => SBBreakpoint(%p)", 
                     target_sp.get(), (uint32_t) bp_id, sb_breakpoint.get());
    }

    return sb_breakpoint;
}

bool
SBTarget::EnableAllBreakpoints ()
{
    TargetSP target_sp(GetSP());
    if (target_sp)
    {
        Mutex::Locker api_locker (target_sp->GetAPIMutex());
        target_sp->EnableAllBreakpoints ();
        return true;
    }
    return false;
}

bool
SBTarget::DisableAllBreakpoints ()
{
    TargetSP target_sp(GetSP());
    if (target_sp)
    {
        Mutex::Locker api_locker (target_sp->GetAPIMutex());
        target_sp->DisableAllBreakpoints ();
        return true;
    }
    return false;
}

bool
SBTarget::DeleteAllBreakpoints ()
{
    TargetSP target_sp(GetSP());
    if (target_sp)
    {
        Mutex::Locker api_locker (target_sp->GetAPIMutex());
        target_sp->RemoveAllBreakpoints ();
        return true;
    }
    return false;
}

uint32_t
SBTarget::GetNumWatchpoints () const
{
    TargetSP target_sp(GetSP());
    if (target_sp)
    {
        // The watchpoint list is thread safe, no need to lock
        return target_sp->GetWatchpointList().GetSize();
    }
    return 0;
}

SBWatchpoint
SBTarget::GetWatchpointAtIndex (uint32_t idx) const
{
    SBWatchpoint sb_watchpoint;
    TargetSP target_sp(GetSP());
    if (target_sp)
    {
        // The watchpoint list is thread safe, no need to lock
        sb_watchpoint.SetSP (target_sp->GetWatchpointList().GetByIndex(idx));
    }
    return sb_watchpoint;
}

bool
SBTarget::DeleteWatchpoint (watch_id_t wp_id)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    bool result = false;
    TargetSP target_sp(GetSP());
    if (target_sp)
    {
        Mutex::Locker api_locker (target_sp->GetAPIMutex());
        Mutex::Locker locker;
        target_sp->GetWatchpointList().GetListMutex(locker);
        result = target_sp->RemoveWatchpointByID (wp_id);
    }

    if (log)
    {
        log->Printf ("SBTarget(%p)::WatchpointDelete (wp_id=%d) => %i", target_sp.get(), (uint32_t) wp_id, result);
    }

    return result;
}

SBWatchpoint
SBTarget::FindWatchpointByID (lldb::watch_id_t wp_id)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBWatchpoint sb_watchpoint;
    lldb::WatchpointSP watchpoint_sp;
    TargetSP target_sp(GetSP());
    if (target_sp && wp_id != LLDB_INVALID_WATCH_ID)
    {
        Mutex::Locker api_locker (target_sp->GetAPIMutex());
        Mutex::Locker locker;
        target_sp->GetWatchpointList().GetListMutex(locker);
        watchpoint_sp = target_sp->GetWatchpointList().FindByID(wp_id);
        sb_watchpoint.SetSP (watchpoint_sp);
    }

    if (log)
    {
        log->Printf ("SBTarget(%p)::FindWatchpointByID (bp_id=%d) => SBWatchpoint(%p)", 
                     target_sp.get(), (uint32_t) wp_id, watchpoint_sp.get());
    }

    return sb_watchpoint;
}

lldb::SBWatchpoint
SBTarget::WatchAddress (lldb::addr_t addr, size_t size, bool read, bool write, SBError &error)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    
    SBWatchpoint sb_watchpoint;
    lldb::WatchpointSP watchpoint_sp;
    TargetSP target_sp(GetSP());
    if (target_sp && (read || write) && addr != LLDB_INVALID_ADDRESS && size > 0)
    {
        Mutex::Locker api_locker (target_sp->GetAPIMutex());
        uint32_t watch_type = 0;
        if (read)
            watch_type |= LLDB_WATCH_TYPE_READ;
        if (write)
            watch_type |= LLDB_WATCH_TYPE_WRITE;
        // Target::CreateWatchpoint() is thread safe.
        Error cw_error;
        // This API doesn't take in a type, so we can't figure out what it is.
        ClangASTType *type = NULL;
        watchpoint_sp = target_sp->CreateWatchpoint(addr, size, type, watch_type, cw_error);
        error.SetError(cw_error);
        sb_watchpoint.SetSP (watchpoint_sp);
    }
    
    if (log)
    {
        log->Printf ("SBTarget(%p)::WatchAddress (addr=0x%llx, 0x%u) => SBWatchpoint(%p)", 
                     target_sp.get(), addr, (uint32_t) size, watchpoint_sp.get());
    }
    
    return sb_watchpoint;
}

bool
SBTarget::EnableAllWatchpoints ()
{
    TargetSP target_sp(GetSP());
    if (target_sp)
    {
        Mutex::Locker api_locker (target_sp->GetAPIMutex());
        Mutex::Locker locker;
        target_sp->GetWatchpointList().GetListMutex(locker);
        target_sp->EnableAllWatchpoints ();
        return true;
    }
    return false;
}

bool
SBTarget::DisableAllWatchpoints ()
{
    TargetSP target_sp(GetSP());
    if (target_sp)
    {
        Mutex::Locker api_locker (target_sp->GetAPIMutex());
        Mutex::Locker locker;
        target_sp->GetWatchpointList().GetListMutex(locker);
        target_sp->DisableAllWatchpoints ();
        return true;
    }
    return false;
}

bool
SBTarget::DeleteAllWatchpoints ()
{
    TargetSP target_sp(GetSP());
    if (target_sp)
    {
        Mutex::Locker api_locker (target_sp->GetAPIMutex());
        Mutex::Locker locker;
        target_sp->GetWatchpointList().GetListMutex(locker);
        target_sp->RemoveAllWatchpoints ();
        return true;
    }
    return false;
}


lldb::SBModule
SBTarget::AddModule (const char *path,
                     const char *triple,
                     const char *uuid_cstr)
{
    return AddModule (path, triple, uuid_cstr, NULL);
}

lldb::SBModule
SBTarget::AddModule (const char *path,
                     const char *triple,
                     const char *uuid_cstr,
                     const char *symfile)
{
    lldb::SBModule sb_module;
    TargetSP target_sp(GetSP());
    if (target_sp)
    {
        ModuleSpec module_spec;
        if (path)
            module_spec.GetFileSpec().SetFile(path, false);
        
        if (uuid_cstr)
            module_spec.GetUUID().SetFromCString(uuid_cstr);
        
        if (triple)
            module_spec.GetArchitecture().SetTriple (triple, target_sp->GetPlatform ().get());
        
        if (symfile)
            module_spec.GetSymbolFileSpec ().SetFile(symfile, false);

        sb_module.SetSP(target_sp->GetSharedModule (module_spec));
    }
    return sb_module;
}

bool
SBTarget::AddModule (lldb::SBModule &module)
{
    TargetSP target_sp(GetSP());
    if (target_sp)
    {
        target_sp->GetImages().AppendIfNeeded (module.GetSP());
        return true;
    }
    return false;
}

uint32_t
SBTarget::GetNumModules () const
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    uint32_t num = 0;
    TargetSP target_sp(GetSP());
    if (target_sp)
    {
        // The module list is thread safe, no need to lock
        num = target_sp->GetImages().GetSize();
    }

    if (log)
        log->Printf ("SBTarget(%p)::GetNumModules () => %d", target_sp.get(), num);

    return num;
}

void
SBTarget::Clear ()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    if (log)
        log->Printf ("SBTarget(%p)::Clear ()", m_opaque_sp.get());

    m_opaque_sp.reset();
}


SBModule
SBTarget::FindModule (const SBFileSpec &sb_file_spec)
{
    SBModule sb_module;
    TargetSP target_sp(GetSP());
    if (target_sp && sb_file_spec.IsValid())
    {
        ModuleSpec module_spec(*sb_file_spec);
        // The module list is thread safe, no need to lock
        sb_module.SetSP (target_sp->GetImages().FindFirstModule (module_spec));
    }
    return sb_module;
}

lldb::ByteOrder
SBTarget::GetByteOrder ()
{
    TargetSP target_sp(GetSP());
    if (target_sp)
        return target_sp->GetArchitecture().GetByteOrder();
    return eByteOrderInvalid;
}

const char *
SBTarget::GetTriple ()
{
    TargetSP target_sp(GetSP());
    if (target_sp)
    {
        std::string triple (target_sp->GetArchitecture().GetTriple().str());
        // Unique the string so we don't run into ownership issues since
        // the const strings put the string into the string pool once and
        // the strings never comes out
        ConstString const_triple (triple.c_str());
        return const_triple.GetCString();
    }
    return NULL;
}

uint32_t
SBTarget::GetAddressByteSize()
{
    TargetSP target_sp(GetSP());
    if (target_sp)
        return target_sp->GetArchitecture().GetAddressByteSize();
    return sizeof(void*);
}


SBModule
SBTarget::GetModuleAtIndex (uint32_t idx)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBModule sb_module;
    ModuleSP module_sp;
    TargetSP target_sp(GetSP());
    if (target_sp)
    {
        // The module list is thread safe, no need to lock
        module_sp = target_sp->GetImages().GetModuleAtIndex(idx);
        sb_module.SetSP (module_sp);
    }

    if (log)
    {
        log->Printf ("SBTarget(%p)::GetModuleAtIndex (idx=%d) => SBModule(%p)", 
                     target_sp.get(), idx, module_sp.get());
    }

    return sb_module;
}

bool
SBTarget::RemoveModule (lldb::SBModule module)
{
    TargetSP target_sp(GetSP());
    if (target_sp)
        return target_sp->GetImages().Remove(module.GetSP());
    return false;
}


SBBroadcaster
SBTarget::GetBroadcaster () const
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    TargetSP target_sp(GetSP());
    SBBroadcaster broadcaster(target_sp.get(), false);
    
    if (log)
        log->Printf ("SBTarget(%p)::GetBroadcaster () => SBBroadcaster(%p)", 
                     target_sp.get(), broadcaster.get());

    return broadcaster;
}

bool
SBTarget::GetDescription (SBStream &description, lldb::DescriptionLevel description_level)
{
    Stream &strm = description.ref();

    TargetSP target_sp(GetSP());
    if (target_sp)
    {
        target_sp->Dump (&strm, description_level);
    }
    else
        strm.PutCString ("No value");
    
    return true;
}

lldb::SBSymbolContextList
SBTarget::FindFunctions (const char *name, uint32_t name_type_mask)
{
    lldb::SBSymbolContextList sb_sc_list;
    if (name && name[0])
    {
        TargetSP target_sp(GetSP());
        if (target_sp)
        {
            const bool symbols_ok = true;
            const bool inlines_ok = true;
            const bool append = true;
            target_sp->GetImages().FindFunctions (ConstString(name), 
                                                  name_type_mask, 
                                                  symbols_ok,
                                                  inlines_ok,
                                                  append, 
                                                  *sb_sc_list);
        }
    }
    return sb_sc_list;
}

lldb::SBType
SBTarget::FindFirstType (const char* type)
{
    TargetSP target_sp(GetSP());
    if (type && target_sp)
    {
        size_t count = target_sp->GetImages().GetSize();
        for (size_t idx = 0; idx < count; idx++)
        {
            SBType found_at_idx = GetModuleAtIndex(idx).FindFirstType(type);
            
            if (found_at_idx.IsValid())
                return found_at_idx;
        }
    }
    return SBType();
}

lldb::SBTypeList
SBTarget::FindTypes (const char* type)
{
    
    SBTypeList retval;
    
    TargetSP target_sp(GetSP());
    if (type && target_sp)
    {
        ModuleList& images = target_sp->GetImages();
        ConstString name_const(type);
        bool exact_match = false;
        SymbolContext sc;
        TypeList type_list;
        
        uint32_t num_matches = images.FindTypes (sc,
                                                 name_const,
                                                 exact_match,
                                                 UINT32_MAX,
                                                 type_list);
        
        for (size_t idx = 0; idx < num_matches; idx++)
        {
            TypeSP type_sp (type_list.GetTypeAtIndex(idx));
            if (type_sp)
                retval.Append(SBType(type_sp));
        }
    }
    return retval;
}

SBValueList
SBTarget::FindGlobalVariables (const char *name, uint32_t max_matches)
{
    SBValueList sb_value_list;
    
    TargetSP target_sp(GetSP());
    if (name && target_sp)
    {
        VariableList variable_list;
        const bool append = true;
        const uint32_t match_count = target_sp->GetImages().FindGlobalVariables (ConstString (name), 
                                                                                 append, 
                                                                                 max_matches,
                                                                                 variable_list);
        
        if (match_count > 0)
        {
            ExecutionContextScope *exe_scope = target_sp->GetProcessSP().get();
            if (exe_scope == NULL)
                exe_scope = target_sp.get();
            ValueObjectList &value_object_list = sb_value_list.ref();
            for (uint32_t i=0; i<match_count; ++i)
            {
                lldb::ValueObjectSP valobj_sp (ValueObjectVariable::Create (exe_scope, variable_list.GetVariableAtIndex(i)));
                if (valobj_sp)
                    value_object_list.Append(valobj_sp);
            }
        }
    }

    return sb_value_list;
}

SBSourceManager
SBTarget::GetSourceManager()
{
    SBSourceManager source_manager (*this);
    return source_manager;
}

lldb::SBInstructionList
SBTarget::ReadInstructions (lldb::SBAddress base_addr, uint32_t count)
{
    SBInstructionList sb_instructions;
    
    TargetSP target_sp(GetSP());
    if (target_sp)
    {
        Address *addr_ptr = base_addr.get();
        
        if (addr_ptr)
        {
            DataBufferHeap data (target_sp->GetArchitecture().GetMaximumOpcodeByteSize() * count, 0);
            bool prefer_file_cache = false;
            lldb_private::Error error;
            const size_t bytes_read = target_sp->ReadMemory(*addr_ptr, prefer_file_cache, data.GetBytes(), data.GetByteSize(), error);
            sb_instructions.SetDisassembler (Disassembler::DisassembleBytes (target_sp->GetArchitecture(),
                                                                             NULL,
                                                                             *addr_ptr,
                                                                             data.GetBytes(),
                                                                             bytes_read,
                                                                             count));
        }
    }
    
    return sb_instructions;
    
}

lldb::SBInstructionList
SBTarget::GetInstructions (lldb::SBAddress base_addr, const void *buf, size_t size)
{
    SBInstructionList sb_instructions;
    
    TargetSP target_sp(GetSP());
    if (target_sp)
    {
        Address addr;
        
        if (base_addr.get())
            addr = *base_addr.get();
        
        sb_instructions.SetDisassembler (Disassembler::DisassembleBytes (target_sp->GetArchitecture(),
                                                                         NULL,
                                                                         addr,
                                                                         buf,
                                                                         size));
    }

    return sb_instructions;
}

lldb::SBInstructionList
SBTarget::GetInstructions (lldb::addr_t base_addr, const void *buf, size_t size)
{
    return GetInstructions (ResolveLoadAddress(base_addr), buf, size);
}

SBError
SBTarget::SetSectionLoadAddress (lldb::SBSection section,
                                 lldb::addr_t section_base_addr)
{
    SBError sb_error;
    TargetSP target_sp(GetSP());
    if (target_sp)
    {
        if (!section.IsValid())
        {
            sb_error.SetErrorStringWithFormat ("invalid section");
        }
        else
        {
            SectionSP section_sp (section.GetSP());
            if (section_sp)
            {
                if (section_sp->IsThreadSpecific())
                {
                    sb_error.SetErrorString ("thread specific sections are not yet supported");
                }
                else
                {
                    target_sp->GetSectionLoadList().SetSectionLoadAddress (section_sp, section_base_addr);
                }
            }
        }
    }
    else
    {
        sb_error.SetErrorString ("invalid target");
    }
    return sb_error;
}

SBError
SBTarget::ClearSectionLoadAddress (lldb::SBSection section)
{
    SBError sb_error;
    
    TargetSP target_sp(GetSP());
    if (target_sp)
    {
        if (!section.IsValid())
        {
            sb_error.SetErrorStringWithFormat ("invalid section");
        }
        else
        {
            target_sp->GetSectionLoadList().SetSectionUnloaded (section.GetSP());
        }
    }
    else
    {
        sb_error.SetErrorStringWithFormat ("invalid target");
    }
    return sb_error;
}

SBError
SBTarget::SetModuleLoadAddress (lldb::SBModule module, int64_t slide_offset)
{
    SBError sb_error;
    
    TargetSP target_sp(GetSP());
    if (target_sp)
    {
        ModuleSP module_sp (module.GetSP());
        if (module_sp)
        {
            bool changed = false;
            if (module_sp->SetLoadAddress (*target_sp, slide_offset, changed))
            {
                // The load was successful, make sure that at least some sections
                // changed before we notify that our module was loaded.
                if (changed)
                {
                    ModuleList module_list;
                    module_list.Append(module_sp);
                    target_sp->ModulesDidLoad (module_list);
                }
            }
        }
        else
        {
            sb_error.SetErrorStringWithFormat ("invalid module");
        }

    }
    else
    {
        sb_error.SetErrorStringWithFormat ("invalid target");
    }
    return sb_error;
}

SBError
SBTarget::ClearModuleLoadAddress (lldb::SBModule module)
{
    SBError sb_error;
    
    char path[PATH_MAX];
    TargetSP target_sp(GetSP());
    if (target_sp)
    {
        ModuleSP module_sp (module.GetSP());
        if (module_sp)
        {
            ObjectFile *objfile = module_sp->GetObjectFile();
            if (objfile)
            {
                SectionList *section_list = objfile->GetSectionList();
                if (section_list)
                {
                    const size_t num_sections = section_list->GetSize();
                    for (size_t sect_idx = 0; sect_idx < num_sections; ++sect_idx)
                    {
                        SectionSP section_sp (section_list->GetSectionAtIndex(sect_idx));
                        if (section_sp)
                            target_sp->GetSectionLoadList().SetSectionUnloaded (section_sp);
                    }
                }
                else
                {
                    module_sp->GetFileSpec().GetPath (path, sizeof(path));
                    sb_error.SetErrorStringWithFormat ("no sections in object file '%s'", path);
                }
            }
            else
            {
                module_sp->GetFileSpec().GetPath (path, sizeof(path));
                sb_error.SetErrorStringWithFormat ("no object file for module '%s'", path);
            }
        }
        else
        {
            sb_error.SetErrorStringWithFormat ("invalid module");
        }
    }
    else
    {
        sb_error.SetErrorStringWithFormat ("invalid target");
    }
    return sb_error;
}


