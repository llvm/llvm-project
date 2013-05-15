//===-- SBProcess.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBProcess_h_
#define LLDB_SBProcess_h_

#include "lldb/API/SBDefines.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBTarget.h"
#include <stdio.h>

namespace lldb {

class SBEvent;

class SBProcess
{
public:
    //------------------------------------------------------------------
    /// Broadcaster event bits definitions.
    //------------------------------------------------------------------
    enum
    {
        eBroadcastBitStateChanged   = (1 << 0),
        eBroadcastBitInterrupt      = (1 << 1),
        eBroadcastBitSTDOUT         = (1 << 2),
        eBroadcastBitSTDERR         = (1 << 3),
        eBroadcastBitProfileData    = (1 << 4)
    };

    SBProcess ();

    SBProcess (const lldb::SBProcess& rhs);

    const lldb::SBProcess&
    operator = (const lldb::SBProcess& rhs);

    SBProcess (const lldb::ProcessSP &process_sp);
    
    ~SBProcess();

    static const char *
    GetBroadcasterClassName ();
    
    const char *
    GetPluginName ();
    
    // DEPRECATED: use GetPluginName()
    const char *
    GetShortPluginName ();
    
    void
    Clear ();

    bool
    IsValid() const;

    lldb::SBTarget
    GetTarget() const;

    lldb::ByteOrder
    GetByteOrder() const;

    size_t
    PutSTDIN (const char *src, size_t src_len);

    size_t
    GetSTDOUT (char *dst, size_t dst_len) const;

    size_t
    GetSTDERR (char *dst, size_t dst_len) const;

    size_t
    GetAsyncProfileData(char *dst, size_t dst_len) const;
    
    void
    ReportEventState (const lldb::SBEvent &event, FILE *out) const;

    void
    AppendEventStateReport (const lldb::SBEvent &event, lldb::SBCommandReturnObject &result);

    //------------------------------------------------------------------
    /// Remote connection related functions. These will fail if the
    /// process is not in eStateConnected. They are intended for use
    /// when connecting to an externally managed debugserver instance.
    //------------------------------------------------------------------
    bool
    RemoteAttachToProcessWithID (lldb::pid_t pid,
                                 lldb::SBError& error);
    
    bool
    RemoteLaunch (char const **argv,
                  char const **envp,
                  const char *stdin_path,
                  const char *stdout_path,
                  const char *stderr_path,
                  const char *working_directory,
                  uint32_t launch_flags,
                  bool stop_at_entry,
                  lldb::SBError& error);
    
    //------------------------------------------------------------------
    // Thread related functions
    //------------------------------------------------------------------
    uint32_t
    GetNumThreads ();

    lldb::SBThread
    GetThreadAtIndex (size_t index);

    lldb::SBThread
    GetThreadByID (lldb::tid_t sb_thread_id);

    lldb::SBThread
    GetThreadByIndexID (uint32_t index_id);

    lldb::SBThread
    GetSelectedThread () const;

    //------------------------------------------------------------------
    // Function for lazily creating a thread using the current OS
    // plug-in. This function will be removed in the future when there
    // are APIs to create SBThread objects through the interface and add
    // them to the process through the SBProcess API.
    //------------------------------------------------------------------
    lldb::SBThread
    CreateOSPluginThread (lldb::tid_t tid, lldb::addr_t context);

    bool
    SetSelectedThread (const lldb::SBThread &thread);

    bool
    SetSelectedThreadByID (lldb::tid_t tid);
    
    bool
    SetSelectedThreadByIndexID (uint32_t index_id);

    //------------------------------------------------------------------
    // Stepping related functions
    //------------------------------------------------------------------

    lldb::StateType
    GetState ();

    int
    GetExitStatus ();

    const char *
    GetExitDescription ();

    //------------------------------------------------------------------
    /// Gets the process ID
    ///
    /// Returns the process identifier for the process as it is known
    /// on the system on which the process is running. For unix systems
    /// this is typically the same as if you called "getpid()" in the
    /// process.
    ///
    /// @return
    ///     Returns LLDB_INVALID_PROCESS_ID if this object does not
    ///     contain a valid process object, or if the process has not
    ///     been launched. Returns a valid process ID if the process is
    ///     valid.
    //------------------------------------------------------------------
    lldb::pid_t
    GetProcessID ();

    //------------------------------------------------------------------
    /// Gets the unique ID associated with this process object
    ///
    /// Unique IDs start at 1 and increment up with each new process
    /// instance. Since starting a process on a system might always
    /// create a process with the same process ID, there needs to be a
    /// way to tell two process instances apart.
    ///
    /// @return
    ///     Returns a non-zero integer ID if this object contains a
    ///     valid process object, zero if this object does not contain
    ///     a valid process object.
    //------------------------------------------------------------------
    uint32_t
    GetUniqueID();

    uint32_t
    GetAddressByteSize() const;

    lldb::SBError
    Destroy ();

    lldb::SBError
    Continue ();

    lldb::SBError
    Stop ();

    lldb::SBError
    Kill ();

    lldb::SBError
    Detach ();

    lldb::SBError
    Detach (bool keep_stopped);

    lldb::SBError
    Signal (int signal);

    void
    SendAsyncInterrupt();
    
    uint32_t
    GetStopID(bool include_expression_stops = false);
    
    size_t
    ReadMemory (addr_t addr, void *buf, size_t size, lldb::SBError &error);

    size_t
    WriteMemory (addr_t addr, const void *buf, size_t size, lldb::SBError &error);

    size_t
    ReadCStringFromMemory (addr_t addr, void *buf, size_t size, lldb::SBError &error);

    uint64_t
    ReadUnsignedFromMemory (addr_t addr, uint32_t byte_size, lldb::SBError &error);

    lldb::addr_t
    ReadPointerFromMemory (addr_t addr, lldb::SBError &error);

    // Events
    static lldb::StateType
    GetStateFromEvent (const lldb::SBEvent &event);

    static bool
    GetRestartedFromEvent (const lldb::SBEvent &event);
    
    static size_t
    GetNumRestartedReasonsFromEvent (const lldb::SBEvent &event);
    
    static const char *
    GetRestartedReasonAtIndexFromEvent (const lldb::SBEvent &event, size_t idx);

    static lldb::SBProcess
    GetProcessFromEvent (const lldb::SBEvent &event);
    
    static bool
    EventIsProcessEvent (const lldb::SBEvent &event);

    lldb::SBBroadcaster
    GetBroadcaster () const;

    static const char *
    GetBroadcasterClass ();

    bool
    GetDescription (lldb::SBStream &description);

    uint32_t
    GetNumSupportedHardwareWatchpoints (lldb::SBError &error) const;

    uint32_t
    LoadImage (lldb::SBFileSpec &image_spec, lldb::SBError &error);
    
    lldb::SBError
    UnloadImage (uint32_t image_token);
    
protected:
    friend class SBAddress;
    friend class SBBreakpoint;
    friend class SBBreakpointLocation;
    friend class SBCommandInterpreter;
    friend class SBDebugger;
    friend class SBFunction;
    friend class SBModule;
    friend class SBTarget;
    friend class SBThread;
    friend class SBValue;

    lldb::ProcessSP
    GetSP() const;
    
    void
    SetSP (const lldb::ProcessSP &process_sp);

    lldb::ProcessWP m_opaque_wp;
};

}  // namespace lldb

#endif  // LLDB_SBProcess_h_
