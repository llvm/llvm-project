//===-- SBProcess.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBProcess.h"

#include "lldb/lldb-defines.h"
#include "lldb/lldb-types.h"

#include "lldb/Interpreter/Args.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/State.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

// Project includes

#include "lldb/API/SBBroadcaster.h"
#include "lldb/API/SBCommandReturnObject.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBEvent.h"
#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBThread.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBStringList.h"

using namespace lldb;
using namespace lldb_private;


SBProcess::SBProcess () :
    m_opaque_wp()
{
}


//----------------------------------------------------------------------
// SBProcess constructor
//----------------------------------------------------------------------

SBProcess::SBProcess (const SBProcess& rhs) :
    m_opaque_wp (rhs.m_opaque_wp)
{
}


SBProcess::SBProcess (const lldb::ProcessSP &process_sp) :
    m_opaque_wp (process_sp)
{
}

const SBProcess&
SBProcess::operator = (const SBProcess& rhs)
{
    if (this != &rhs)
        m_opaque_wp = rhs.m_opaque_wp;
    return *this;
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
SBProcess::~SBProcess()
{
}

const char *
SBProcess::GetBroadcasterClassName ()
{
    return Process::GetStaticBroadcasterClass().AsCString();
}

const char *
SBProcess::GetPluginName ()
{
    ProcessSP process_sp(GetSP());
    if (process_sp)
    {
        return process_sp->GetPluginName();
    }
    return "<Unknown>";
}

const char *
SBProcess::GetShortPluginName ()
{
    ProcessSP process_sp(GetSP());
    if (process_sp)
    {
        return process_sp->GetShortPluginName();
    }
    return "<Unknown>";
}


lldb::ProcessSP
SBProcess::GetSP() const
{
    return m_opaque_wp.lock();
}

void
SBProcess::SetSP (const ProcessSP &process_sp)
{
    m_opaque_wp = process_sp;
}

void
SBProcess::Clear ()
{
    m_opaque_wp.reset();
}


bool
SBProcess::IsValid() const
{
    ProcessSP process_sp(m_opaque_wp.lock());
    return ((bool) process_sp && process_sp->IsValid());
}

bool
SBProcess::RemoteLaunch (char const **argv,
                         char const **envp,
                         const char *stdin_path,
                         const char *stdout_path,
                         const char *stderr_path,
                         const char *working_directory,
                         uint32_t launch_flags,
                         bool stop_at_entry,
                         lldb::SBError& error)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log) {
        log->Printf ("SBProcess(%p)::RemoteLaunch (argv=%p, envp=%p, stdin=%s, stdout=%s, stderr=%s, working-dir=%s, launch_flags=0x%x, stop_at_entry=%i, &error (%p))...",
                     m_opaque_wp.lock().get(),
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
    
    ProcessSP process_sp(GetSP());
    if (process_sp)
    {
        Mutex::Locker api_locker (process_sp->GetTarget().GetAPIMutex());
        if (process_sp->GetState() == eStateConnected)
        {
            if (stop_at_entry)
                launch_flags |= eLaunchFlagStopAtEntry;
            ProcessLaunchInfo launch_info (stdin_path, 
                                           stdout_path,
                                           stderr_path,
                                           working_directory,
                                           launch_flags);
            Module *exe_module = process_sp->GetTarget().GetExecutableModulePointer();
            if (exe_module)
                launch_info.SetExecutableFile(exe_module->GetFileSpec(), true);
            if (argv)
                launch_info.GetArguments().AppendArguments (argv);
            if (envp)
                launch_info.GetEnvironmentEntries ().SetArguments (envp);
            error.SetError (process_sp->Launch (launch_info));
        }
        else
        {
            error.SetErrorString ("must be in eStateConnected to call RemoteLaunch");
        }
    }
    else
    {
        error.SetErrorString ("unable to attach pid");
    }
    
    if (log) {
        SBStream sstr;
        error.GetDescription (sstr);
        log->Printf ("SBProcess(%p)::RemoteLaunch (...) => SBError (%p): %s", process_sp.get(), error.get(), sstr.GetData());
    }
    
    return error.Success();
}

bool
SBProcess::RemoteAttachToProcessWithID (lldb::pid_t pid, lldb::SBError& error)
{
    ProcessSP process_sp(GetSP());
    if (process_sp)
    {
        Mutex::Locker api_locker (process_sp->GetTarget().GetAPIMutex());
        if (process_sp->GetState() == eStateConnected)
        {
            ProcessAttachInfo attach_info;
            attach_info.SetProcessID (pid);
            error.SetError (process_sp->Attach (attach_info));            
        }
        else
        {
            error.SetErrorString ("must be in eStateConnected to call RemoteAttachToProcessWithID");
        }
    }
    else
    {
        error.SetErrorString ("unable to attach pid");
    }

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log) {
        SBStream sstr;
        error.GetDescription (sstr);
        log->Printf ("SBProcess(%p)::RemoteAttachToProcessWithID (%llu) => SBError (%p): %s", process_sp.get(), pid, error.get(), sstr.GetData());
    }

    return error.Success();
}


uint32_t
SBProcess::GetNumThreads ()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    uint32_t num_threads = 0;
    ProcessSP process_sp(GetSP());
    if (process_sp)
    {
        Process::StopLocker stop_locker;
        
        const bool can_update = stop_locker.TryLock(&process_sp->GetRunLock());
        Mutex::Locker api_locker (process_sp->GetTarget().GetAPIMutex());
        num_threads = process_sp->GetThreadList().GetSize(can_update);
    }

    if (log)
        log->Printf ("SBProcess(%p)::GetNumThreads () => %d", process_sp.get(), num_threads);

    return num_threads;
}

SBThread
SBProcess::GetSelectedThread () const
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBThread sb_thread;
    ThreadSP thread_sp;
    ProcessSP process_sp(GetSP());
    if (process_sp)
    {
        Mutex::Locker api_locker (process_sp->GetTarget().GetAPIMutex());
        thread_sp = process_sp->GetThreadList().GetSelectedThread();
        sb_thread.SetThread (thread_sp);
    }

    if (log)
    {
        log->Printf ("SBProcess(%p)::GetSelectedThread () => SBThread(%p)", process_sp.get(), thread_sp.get());
    }

    return sb_thread;
}

SBTarget
SBProcess::GetTarget() const
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBTarget sb_target;
    TargetSP target_sp;
    ProcessSP process_sp(GetSP());
    if (process_sp)
    {
        target_sp = process_sp->GetTarget().shared_from_this();
        sb_target.SetSP (target_sp);
    }
    
    if (log)
        log->Printf ("SBProcess(%p)::GetTarget () => SBTarget(%p)", process_sp.get(), target_sp.get());

    return sb_target;
}


size_t
SBProcess::PutSTDIN (const char *src, size_t src_len)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    size_t ret_val = 0;
    ProcessSP process_sp(GetSP());
    if (process_sp)
    {
        Error error;
        ret_val =  process_sp->PutSTDIN (src, src_len, error);
    }
    
    if (log)
        log->Printf ("SBProcess(%p)::PutSTDIN (src=\"%s\", src_len=%d) => %lu", 
                     process_sp.get(), 
                     src, 
                     (uint32_t) src_len, 
                     ret_val);

    return ret_val;
}

size_t
SBProcess::GetSTDOUT (char *dst, size_t dst_len) const
{
    size_t bytes_read = 0;
    ProcessSP process_sp(GetSP());
    if (process_sp)
    {
        Error error;
        bytes_read = process_sp->GetSTDOUT (dst, dst_len, error);
    }
    
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBProcess(%p)::GetSTDOUT (dst=\"%.*s\", dst_len=%llu) => %llu", 
                     process_sp.get(),
                     (int) bytes_read,
                     dst,
                     (uint64_t)dst_len,
                     (uint64_t)bytes_read);

    return bytes_read;
}

size_t
SBProcess::GetSTDERR (char *dst, size_t dst_len) const
{
    size_t bytes_read = 0;
    ProcessSP process_sp(GetSP());
    if (process_sp)
    {
        Error error;
        bytes_read = process_sp->GetSTDERR (dst, dst_len, error);
    }

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBProcess(%p)::GetSTDERR (dst=\"%.*s\", dst_len=%llu) => %llu",
                     process_sp.get(),
                     (int) bytes_read,
                     dst,
                     (uint64_t)dst_len,
                     (uint64_t)bytes_read);

    return bytes_read;
}

void
SBProcess::ReportEventState (const SBEvent &event, FILE *out) const
{
    if (out == NULL)
        return;

    ProcessSP process_sp(GetSP());
    if (process_sp)
    {
        const StateType event_state = SBProcess::GetStateFromEvent (event);
        char message[1024];
        int message_len = ::snprintf (message,
                                      sizeof (message),
                                      "Process %llu %s\n",
                                      process_sp->GetID(),
                                      SBDebugger::StateAsCString (event_state));

        if (message_len > 0)
            ::fwrite (message, 1, message_len, out);
    }
}

void
SBProcess::AppendEventStateReport (const SBEvent &event, SBCommandReturnObject &result)
{
    ProcessSP process_sp(GetSP());
    if (process_sp)
    {
        const StateType event_state = SBProcess::GetStateFromEvent (event);
        char message[1024];
        ::snprintf (message,
                    sizeof (message),
                    "Process %llu %s\n",
                    process_sp->GetID(),
                    SBDebugger::StateAsCString (event_state));

        result.AppendMessage (message);
    }
}

bool
SBProcess::SetSelectedThread (const SBThread &thread)
{
    ProcessSP process_sp(GetSP());
    if (process_sp)
    {
        Mutex::Locker api_locker (process_sp->GetTarget().GetAPIMutex());
        return process_sp->GetThreadList().SetSelectedThreadByID (thread.GetThreadID());
    }
    return false;
}

bool
SBProcess::SetSelectedThreadByID (lldb::tid_t tid)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    bool ret_val = false;
    ProcessSP process_sp(GetSP());
    if (process_sp)
    {
        Mutex::Locker api_locker (process_sp->GetTarget().GetAPIMutex());
        ret_val = process_sp->GetThreadList().SetSelectedThreadByID (tid);
    }

    if (log)
        log->Printf ("SBProcess(%p)::SetSelectedThreadByID (tid=0x%4.4llx) => %s", 
                     process_sp.get(), tid, (ret_val ? "true" : "false"));

    return ret_val;
}

bool
SBProcess::SetSelectedThreadByIndexID (uint32_t index_id)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    bool ret_val = false;
    ProcessSP process_sp(GetSP());
    if (process_sp)
    {
        Mutex::Locker api_locker (process_sp->GetTarget().GetAPIMutex());
        ret_val = process_sp->GetThreadList().SetSelectedThreadByIndexID (index_id);
    }

    if (log)
        log->Printf ("SBProcess(%p)::SetSelectedThreadByID (tid=0x%x) => %s", 
                     process_sp.get(), index_id, (ret_val ? "true" : "false"));

    return ret_val;
}

SBThread
SBProcess::GetThreadAtIndex (size_t index)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBThread sb_thread;
    ThreadSP thread_sp;
    ProcessSP process_sp(GetSP());
    if (process_sp)
    {
        Process::StopLocker stop_locker;
        const bool can_update = stop_locker.TryLock(&process_sp->GetRunLock());
        Mutex::Locker api_locker (process_sp->GetTarget().GetAPIMutex());
        thread_sp = process_sp->GetThreadList().GetThreadAtIndex(index, can_update);
        sb_thread.SetThread (thread_sp);
    }

    if (log)
    {
        log->Printf ("SBProcess(%p)::GetThreadAtIndex (index=%d) => SBThread(%p)",
                     process_sp.get(), (uint32_t) index, thread_sp.get());
    }

    return sb_thread;
}

StateType
SBProcess::GetState ()
{

    StateType ret_val = eStateInvalid;
    ProcessSP process_sp(GetSP());
    if (process_sp)
    {
        Mutex::Locker api_locker (process_sp->GetTarget().GetAPIMutex());
        ret_val = process_sp->GetState();
    }

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBProcess(%p)::GetState () => %s", 
                     process_sp.get(),
                     lldb_private::StateAsCString (ret_val));

    return ret_val;
}


int
SBProcess::GetExitStatus ()
{
    int exit_status = 0;
    ProcessSP process_sp(GetSP());
    if (process_sp)
    {
        Mutex::Locker api_locker (process_sp->GetTarget().GetAPIMutex());
        exit_status = process_sp->GetExitStatus ();
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBProcess(%p)::GetExitStatus () => %i (0x%8.8x)", 
                     process_sp.get(), exit_status, exit_status);

    return exit_status;
}

const char *
SBProcess::GetExitDescription ()
{
    const char *exit_desc = NULL;
    ProcessSP process_sp(GetSP());
    if (process_sp)
    {
        Mutex::Locker api_locker (process_sp->GetTarget().GetAPIMutex());
        exit_desc = process_sp->GetExitDescription ();
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBProcess(%p)::GetExitDescription () => %s", 
                     process_sp.get(), exit_desc);
    return exit_desc;
}

lldb::pid_t
SBProcess::GetProcessID ()
{
    lldb::pid_t ret_val = LLDB_INVALID_PROCESS_ID;
    ProcessSP process_sp(GetSP());
    if (process_sp)
        ret_val = process_sp->GetID();

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBProcess(%p)::GetProcessID () => %llu", process_sp.get(), ret_val);

    return ret_val;
}

ByteOrder
SBProcess::GetByteOrder () const
{
    ByteOrder byteOrder = eByteOrderInvalid;
    ProcessSP process_sp(GetSP());
    if (process_sp)
        byteOrder = process_sp->GetTarget().GetArchitecture().GetByteOrder();
    
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBProcess(%p)::GetByteOrder () => %d", process_sp.get(), byteOrder);

    return byteOrder;
}

uint32_t
SBProcess::GetAddressByteSize () const
{
    uint32_t size = 0;
    ProcessSP process_sp(GetSP());
    if (process_sp)
        size =  process_sp->GetTarget().GetArchitecture().GetAddressByteSize();

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBProcess(%p)::GetAddressByteSize () => %d", process_sp.get(), size);

    return size;
}

SBError
SBProcess::Continue ()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    
    SBError sb_error;
    ProcessSP process_sp(GetSP());

    if (log)
        log->Printf ("SBProcess(%p)::Continue ()...", process_sp.get());

    if (process_sp)
    {
        Mutex::Locker api_locker (process_sp->GetTarget().GetAPIMutex());
        
        Error error (process_sp->Resume());
        if (error.Success())
        {
            if (process_sp->GetTarget().GetDebugger().GetAsyncExecution () == false)
            {
                if (log)
                    log->Printf ("SBProcess(%p)::Continue () waiting for process to stop...", process_sp.get());
                process_sp->WaitForProcessToStop (NULL);
            }
        }
        sb_error.SetError(error);
    }
    else
        sb_error.SetErrorString ("SBProcess is invalid");

    if (log)
    {
        SBStream sstr;
        sb_error.GetDescription (sstr);
        log->Printf ("SBProcess(%p)::Continue () => SBError (%p): %s", process_sp.get(), sb_error.get(), sstr.GetData());
    }

    return sb_error;
}


SBError
SBProcess::Destroy ()
{
    SBError sb_error;
    ProcessSP process_sp(GetSP());
    if (process_sp)
    {
        Mutex::Locker api_locker (process_sp->GetTarget().GetAPIMutex());
        sb_error.SetError(process_sp->Destroy());
    }
    else
        sb_error.SetErrorString ("SBProcess is invalid");

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        SBStream sstr;
        sb_error.GetDescription (sstr);
        log->Printf ("SBProcess(%p)::Destroy () => SBError (%p): %s", 
                     process_sp.get(), 
                     sb_error.get(), 
                     sstr.GetData());
    }

    return sb_error;
}


SBError
SBProcess::Stop ()
{
    SBError sb_error;
    ProcessSP process_sp(GetSP());
    if (process_sp)
    {
        Mutex::Locker api_locker (process_sp->GetTarget().GetAPIMutex());
        sb_error.SetError (process_sp->Halt());
    }
    else
        sb_error.SetErrorString ("SBProcess is invalid");
    
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        SBStream sstr;
        sb_error.GetDescription (sstr);
        log->Printf ("SBProcess(%p)::Stop () => SBError (%p): %s", 
                     process_sp.get(), 
                     sb_error.get(),
                     sstr.GetData());
    }

    return sb_error;
}

SBError
SBProcess::Kill ()
{
    SBError sb_error;
    ProcessSP process_sp(GetSP());
    if (process_sp)
    {
        Mutex::Locker api_locker (process_sp->GetTarget().GetAPIMutex());
        sb_error.SetError (process_sp->Destroy());
    }
    else
        sb_error.SetErrorString ("SBProcess is invalid");

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        SBStream sstr;
        sb_error.GetDescription (sstr);
        log->Printf ("SBProcess(%p)::Kill () => SBError (%p): %s", 
                     process_sp.get(), 
                     sb_error.get(),
                     sstr.GetData());
    }

    return sb_error;
}

SBError
SBProcess::Detach ()
{
    SBError sb_error;
    ProcessSP process_sp(GetSP());
    if (process_sp)
    {
        Mutex::Locker api_locker (process_sp->GetTarget().GetAPIMutex());
        sb_error.SetError (process_sp->Detach());
    }
    else
        sb_error.SetErrorString ("SBProcess is invalid");    

    return sb_error;
}

SBError
SBProcess::Signal (int signo)
{
    SBError sb_error;
    ProcessSP process_sp(GetSP());
    if (process_sp)
    {
        Mutex::Locker api_locker (process_sp->GetTarget().GetAPIMutex());
        sb_error.SetError (process_sp->Signal (signo));
    }
    else
        sb_error.SetErrorString ("SBProcess is invalid");    
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        SBStream sstr;
        sb_error.GetDescription (sstr);
        log->Printf ("SBProcess(%p)::Signal (signo=%i) => SBError (%p): %s", 
                     process_sp.get(), 
                     signo,
                     sb_error.get(),
                     sstr.GetData());
    }
    return sb_error;
}

void
SBProcess::SendAsyncInterrupt ()
{
    ProcessSP process_sp(GetSP());
    if (process_sp)
    {
        process_sp->SendAsyncInterrupt ();
    }
}

SBThread
SBProcess::GetThreadByID (tid_t tid)
{
    SBThread sb_thread;
    ThreadSP thread_sp;
    ProcessSP process_sp(GetSP());
    if (process_sp)
    {
        Mutex::Locker api_locker (process_sp->GetTarget().GetAPIMutex());
        Process::StopLocker stop_locker;
        const bool can_update = stop_locker.TryLock(&process_sp->GetRunLock());
        thread_sp = process_sp->GetThreadList().FindThreadByID (tid, can_update);
        sb_thread.SetThread (thread_sp);
    }

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        log->Printf ("SBProcess(%p)::GetThreadByID (tid=0x%4.4llx) => SBThread (%p)", 
                     process_sp.get(), 
                     tid,
                     thread_sp.get());
    }

    return sb_thread;
}

SBThread
SBProcess::GetThreadByIndexID (uint32_t index_id)
{
    SBThread sb_thread;
    ThreadSP thread_sp;
    ProcessSP process_sp(GetSP());
    if (process_sp)
    {
        Mutex::Locker api_locker (process_sp->GetTarget().GetAPIMutex());
        Process::StopLocker stop_locker;
        const bool can_update = stop_locker.TryLock(&process_sp->GetRunLock());
        thread_sp = process_sp->GetThreadList().FindThreadByIndexID (index_id, can_update);
        sb_thread.SetThread (thread_sp);
    }

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        log->Printf ("SBProcess(%p)::GetThreadByID (tid=0x%x) => SBThread (%p)", 
                     process_sp.get(), 
                     index_id,
                     thread_sp.get());
    }

    return sb_thread;
}

StateType
SBProcess::GetStateFromEvent (const SBEvent &event)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    StateType ret_val = Process::ProcessEventData::GetStateFromEvent (event.get());
    
    if (log)
        log->Printf ("SBProcess::GetStateFromEvent (event.sp=%p) => %s", event.get(),
                     lldb_private::StateAsCString (ret_val));

    return ret_val;
}

bool
SBProcess::GetRestartedFromEvent (const SBEvent &event)
{
    return Process::ProcessEventData::GetRestartedFromEvent (event.get());
}

SBProcess
SBProcess::GetProcessFromEvent (const SBEvent &event)
{
    SBProcess process(Process::ProcessEventData::GetProcessFromEvent (event.get()));
    return process;
}

bool
SBProcess::EventIsProcessEvent (const SBEvent &event)
{
    return strcmp (event.GetBroadcasterClass(), SBProcess::GetBroadcasterClass()) == 0;
}

SBBroadcaster
SBProcess::GetBroadcaster () const
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    ProcessSP process_sp(GetSP());

    SBBroadcaster broadcaster(process_sp.get(), false);

    if (log)
        log->Printf ("SBProcess(%p)::GetBroadcaster () => SBBroadcaster (%p)",  process_sp.get(),
                     broadcaster.get());

    return broadcaster;
}

const char *
SBProcess::GetBroadcasterClass ()
{
    return Process::GetStaticBroadcasterClass().AsCString();
}

size_t
SBProcess::ReadMemory (addr_t addr, void *dst, size_t dst_len, SBError &sb_error)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    size_t bytes_read = 0;

    ProcessSP process_sp(GetSP());

    if (log)
    {
        log->Printf ("SBProcess(%p)::ReadMemory (addr=0x%llx, dst=%p, dst_len=%llu, SBError (%p))...",
                     process_sp.get(), 
                     addr, 
                     dst, 
                     (uint64_t)dst_len,
                     sb_error.get());
    }
    
    if (process_sp)
    {
        Process::StopLocker stop_locker;
        if (stop_locker.TryLock(&process_sp->GetRunLock()))
        {
            Mutex::Locker api_locker (process_sp->GetTarget().GetAPIMutex());
            bytes_read = process_sp->ReadMemory (addr, dst, dst_len, sb_error.ref());
        }
        else
        {
            if (log)
                log->Printf ("SBProcess(%p)::ReadMemory() => error: process is running", process_sp.get());
            sb_error.SetErrorString("process is running");
        }
    }
    else
    {
        sb_error.SetErrorString ("SBProcess is invalid");
    }

    if (log)
    {
        SBStream sstr;
        sb_error.GetDescription (sstr);
        log->Printf ("SBProcess(%p)::ReadMemory (addr=0x%llx, dst=%p, dst_len=%llu, SBError (%p): %s) => %llu", 
                     process_sp.get(), 
                     addr, 
                     dst, 
                     (uint64_t)dst_len,
                     sb_error.get(), 
                     sstr.GetData(),
                     (uint64_t)bytes_read);
    }

    return bytes_read;
}

size_t
SBProcess::ReadCStringFromMemory (addr_t addr, void *buf, size_t size, lldb::SBError &sb_error)
{
    size_t bytes_read = 0;
    ProcessSP process_sp(GetSP());
    if (process_sp)
    {
        Process::StopLocker stop_locker;
        if (stop_locker.TryLock(&process_sp->GetRunLock()))
        {
            Mutex::Locker api_locker (process_sp->GetTarget().GetAPIMutex());
            bytes_read = process_sp->ReadCStringFromMemory (addr, (char *)buf, size, sb_error.ref());
        }
        else
        {
            LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
            if (log)
                log->Printf ("SBProcess(%p)::ReadCStringFromMemory() => error: process is running", process_sp.get());
            sb_error.SetErrorString("process is running");
        }
    }
    else
    {
        sb_error.SetErrorString ("SBProcess is invalid");
    }
    return bytes_read;
}

uint64_t
SBProcess::ReadUnsignedFromMemory (addr_t addr, uint32_t byte_size, lldb::SBError &sb_error)
{
    uint64_t value = 0;
    ProcessSP process_sp(GetSP());
    if (process_sp)
    {
        Process::StopLocker stop_locker;
        if (stop_locker.TryLock(&process_sp->GetRunLock()))
        {
            Mutex::Locker api_locker (process_sp->GetTarget().GetAPIMutex());
            value = process_sp->ReadUnsignedIntegerFromMemory (addr, byte_size, 0, sb_error.ref());
        }
        else
        {
            LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
            if (log)
                log->Printf ("SBProcess(%p)::ReadUnsignedFromMemory() => error: process is running", process_sp.get());
            sb_error.SetErrorString("process is running");
        }
    }
    else
    {
        sb_error.SetErrorString ("SBProcess is invalid");
    }
    return value;
}

lldb::addr_t
SBProcess::ReadPointerFromMemory (addr_t addr, lldb::SBError &sb_error)
{
    lldb::addr_t ptr = LLDB_INVALID_ADDRESS;
    ProcessSP process_sp(GetSP());
    if (process_sp)
    {
        Process::StopLocker stop_locker;
        if (stop_locker.TryLock(&process_sp->GetRunLock()))
        {
            Mutex::Locker api_locker (process_sp->GetTarget().GetAPIMutex());
            ptr = process_sp->ReadPointerFromMemory (addr, sb_error.ref());
        }
        else
        {
            LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
            if (log)
                log->Printf ("SBProcess(%p)::ReadPointerFromMemory() => error: process is running", process_sp.get());
            sb_error.SetErrorString("process is running");
        }
    }
    else
    {
        sb_error.SetErrorString ("SBProcess is invalid");
    }
    return ptr;
}

size_t
SBProcess::WriteMemory (addr_t addr, const void *src, size_t src_len, SBError &sb_error)
{
    size_t bytes_written = 0;

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    ProcessSP process_sp(GetSP());

    if (log)
    {
        log->Printf ("SBProcess(%p)::WriteMemory (addr=0x%llx, src=%p, src_len=%llu, SBError (%p))...",
                     process_sp.get(), 
                     addr, 
                     src, 
                     (uint64_t)src_len,
                     sb_error.get());
    }

    if (process_sp)
    {
        Process::StopLocker stop_locker;
        if (stop_locker.TryLock(&process_sp->GetRunLock()))
        {
            Mutex::Locker api_locker (process_sp->GetTarget().GetAPIMutex());
            bytes_written = process_sp->WriteMemory (addr, src, src_len, sb_error.ref());
        }
        else
        {
            if (log)
                log->Printf ("SBProcess(%p)::WriteMemory() => error: process is running", process_sp.get());
            sb_error.SetErrorString("process is running");
        }
    }

    if (log)
    {
        SBStream sstr;
        sb_error.GetDescription (sstr);
        log->Printf ("SBProcess(%p)::WriteMemory (addr=0x%llx, src=%p, src_len=%llu, SBError (%p): %s) => %llu",
                     process_sp.get(), 
                     addr, 
                     src, 
                     (uint64_t)src_len, 
                     sb_error.get(), 
                     sstr.GetData(),
                     (uint64_t)bytes_written);
    }

    return bytes_written;
}

bool
SBProcess::GetDescription (SBStream &description)
{
    Stream &strm = description.ref();

    ProcessSP process_sp(GetSP());
    if (process_sp)
    {
        char path[PATH_MAX];
        GetTarget().GetExecutable().GetPath (path, sizeof(path));
        Module *exe_module = process_sp->GetTarget().GetExecutableModulePointer();
        const char *exe_name = NULL;
        if (exe_module)
            exe_name = exe_module->GetFileSpec().GetFilename().AsCString();

        strm.Printf ("SBProcess: pid = %llu, state = %s, threads = %d%s%s", 
                     process_sp->GetID(),
                     lldb_private::StateAsCString (GetState()), 
                     GetNumThreads(),
                     exe_name ? ", executable = " : "",
                     exe_name ? exe_name : "");
    }
    else
        strm.PutCString ("No value");

    return true;
}

uint32_t
SBProcess::GetNumSupportedHardwareWatchpoints (lldb::SBError &sb_error) const
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    uint32_t num = 0;
    ProcessSP process_sp(GetSP());
    if (process_sp)
    {
        Mutex::Locker api_locker (process_sp->GetTarget().GetAPIMutex());
        sb_error.SetError(process_sp->GetWatchpointSupportInfo (num));
        if (log)
            log->Printf ("SBProcess(%p)::GetNumSupportedHardwareWatchpoints () => %u",
                         process_sp.get(), num);
    }
    else
    {
        sb_error.SetErrorString ("SBProcess is invalid");
    }
    return num;
}

uint32_t
SBProcess::LoadImage (lldb::SBFileSpec &sb_image_spec, lldb::SBError &sb_error)
{
    ProcessSP process_sp(GetSP());
    if (process_sp)
    {
        Process::StopLocker stop_locker;
        if (stop_locker.TryLock(&process_sp->GetRunLock()))
        {
            Mutex::Locker api_locker (process_sp->GetTarget().GetAPIMutex());
            return process_sp->LoadImage (*sb_image_spec, sb_error.ref());
        }
        else
        {
            LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
            if (log)
                log->Printf ("SBProcess(%p)::LoadImage() => error: process is running", process_sp.get());
            sb_error.SetErrorString("process is running");
        }
    }
    return LLDB_INVALID_IMAGE_TOKEN;
}
    
lldb::SBError
SBProcess::UnloadImage (uint32_t image_token)
{
    lldb::SBError sb_error;
    ProcessSP process_sp(GetSP());
    if (process_sp)
    {
        Process::StopLocker stop_locker;
        if (stop_locker.TryLock(&process_sp->GetRunLock()))
        {
            Mutex::Locker api_locker (process_sp->GetTarget().GetAPIMutex());
            sb_error.SetError (process_sp->UnloadImage (image_token));
        }
        else
        {
            LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
            if (log)
                log->Printf ("SBProcess(%p)::UnloadImage() => error: process is running", process_sp.get());
            sb_error.SetErrorString("process is running");
        }
    }
    else
        sb_error.SetErrorString("invalid process");
    return sb_error;
}
