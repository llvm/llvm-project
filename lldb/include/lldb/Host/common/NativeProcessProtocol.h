//===-- NativeProcessProtocol.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_COMMON_NATIVEPROCESSPROTOCOL_H
#define LLDB_HOST_COMMON_NATIVEPROCESSPROTOCOL_H

#include "NativeBreakpointList.h"
#include "NativeThreadProtocol.h"
#include "NativeWatchpointList.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/MainLoop.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/TraceOptions.h"
#include "lldb/Utility/UnimplementedError.h"
#include "lldb/lldb-private-forward.h"
#include "lldb/lldb-types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include <mutex>
#include <unordered_map>
#include <vector>

namespace lldb_private {
class MemoryRegionInfo;
class ResumeActionList;

struct SVR4LibraryInfo {
  std::string name;
  lldb::addr_t link_map;
  lldb::addr_t base_addr;
  lldb::addr_t ld_addr;
  lldb::addr_t next;
};

// NativeProcessProtocol
class NativeProcessProtocol {
public:
  virtual ~NativeProcessProtocol() {}

  virtual Status Resume(const ResumeActionList &resume_actions) = 0;

  virtual Status Halt() = 0;

  virtual Status Detach() = 0;

  /// Sends a process a UNIX signal \a signal.
  ///
  /// \return
  ///     Returns an error object.
  virtual Status Signal(int signo) = 0;

  /// Tells a process to interrupt all operations as if by a Ctrl-C.
  ///
  /// The default implementation will send a local host's equivalent of
  /// a SIGSTOP to the process via the NativeProcessProtocol::Signal()
  /// operation.
  ///
  /// \return
  ///     Returns an error object.
  virtual Status Interrupt();

  virtual Status Kill() = 0;

  // Tells a process not to stop the inferior on given signals and just
  // reinject them back.
  virtual Status IgnoreSignals(llvm::ArrayRef<int> signals);

  // Memory and memory region functions

  virtual Status GetMemoryRegionInfo(lldb::addr_t load_addr,
                                     MemoryRegionInfo &range_info);

  virtual Status ReadMemory(lldb::addr_t addr, void *buf, size_t size,
                            size_t &bytes_read) = 0;

  Status ReadMemoryWithoutTrap(lldb::addr_t addr, void *buf, size_t size,
                               size_t &bytes_read);

  /// Reads a null terminated string from memory.
  ///
  /// Reads up to \p max_size bytes of memory until it finds a '\0'.
  /// If a '\0' is not found then it reads max_size-1 bytes as a string and a
  /// '\0' is added as the last character of the \p buffer.
  ///
  /// \param[in] addr
  ///     The address in memory to read from.
  ///
  /// \param[in] buffer
  ///     An allocated buffer with at least \p max_size size.
  ///
  /// \param[in] max_size
  ///     The maximum number of bytes to read from memory until it reads the
  ///     string.
  ///
  /// \param[out] total_bytes_read
  ///     The number of bytes read from memory into \p buffer.
  ///
  /// \return
  ///     Returns a StringRef backed up by the \p buffer passed in.
  llvm::Expected<llvm::StringRef>
  ReadCStringFromMemory(lldb::addr_t addr, char *buffer, size_t max_size,
                        size_t &total_bytes_read);

  virtual Status WriteMemory(lldb::addr_t addr, const void *buf, size_t size,
                             size_t &bytes_written) = 0;

  virtual llvm::Expected<lldb::addr_t> AllocateMemory(size_t size,
                                                      uint32_t permissions) {
    return llvm::make_error<UnimplementedError>();
  }

  virtual llvm::Error DeallocateMemory(lldb::addr_t addr) {
    return llvm::make_error<UnimplementedError>();
  }

  virtual lldb::addr_t GetSharedLibraryInfoAddress() = 0;

  virtual llvm::Expected<std::vector<SVR4LibraryInfo>>
  GetLoadedSVR4Libraries() {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Not implemented");
  }

  virtual bool IsAlive() const;

  virtual size_t UpdateThreads() = 0;

  virtual const ArchSpec &GetArchitecture() const = 0;

  // Breakpoint functions
  virtual Status SetBreakpoint(lldb::addr_t addr, uint32_t size,
                               bool hardware) = 0;

  virtual Status RemoveBreakpoint(lldb::addr_t addr, bool hardware = false);

  // Hardware Breakpoint functions
  virtual const HardwareBreakpointMap &GetHardwareBreakpointMap() const;

  virtual Status SetHardwareBreakpoint(lldb::addr_t addr, size_t size);

  virtual Status RemoveHardwareBreakpoint(lldb::addr_t addr);

  // Watchpoint functions
  virtual const NativeWatchpointList::WatchpointMap &GetWatchpointMap() const;

  virtual llvm::Optional<std::pair<uint32_t, uint32_t>>
  GetHardwareDebugSupportInfo() const;

  virtual Status SetWatchpoint(lldb::addr_t addr, size_t size,
                               uint32_t watch_flags, bool hardware);

  virtual Status RemoveWatchpoint(lldb::addr_t addr);

  // Accessors
  lldb::pid_t GetID() const { return m_pid; }

  lldb::StateType GetState() const;

  bool IsRunning() const {
    return m_state == lldb::eStateRunning || IsStepping();
  }

  bool IsStepping() const { return m_state == lldb::eStateStepping; }

  bool CanResume() const { return m_state == lldb::eStateStopped; }

  lldb::ByteOrder GetByteOrder() const {
    return GetArchitecture().GetByteOrder();
  }

  uint32_t GetAddressByteSize() const {
    return GetArchitecture().GetAddressByteSize();
  }

  virtual llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  GetAuxvData() const = 0;

  // Exit Status
  virtual llvm::Optional<WaitStatus> GetExitStatus();

  virtual bool SetExitStatus(WaitStatus status, bool bNotifyStateChange);

  // Access to threads
  NativeThreadProtocol *GetThreadAtIndex(uint32_t idx);

  NativeThreadProtocol *GetThreadByID(lldb::tid_t tid);

  void SetCurrentThreadID(lldb::tid_t tid) { m_current_thread_id = tid; }

  lldb::tid_t GetCurrentThreadID() { return m_current_thread_id; }

  NativeThreadProtocol *GetCurrentThread() {
    return GetThreadByID(m_current_thread_id);
  }

  // Access to inferior stdio
  virtual int GetTerminalFileDescriptor() { return m_terminal_fd; }

  // Stop id interface

  uint32_t GetStopID() const;

  // Callbacks for low-level process state changes
  class NativeDelegate {
  public:
    virtual ~NativeDelegate() {}

    virtual void InitializeDelegate(NativeProcessProtocol *process) = 0;

    virtual void ProcessStateChanged(NativeProcessProtocol *process,
                                     lldb::StateType state) = 0;

    virtual void DidExec(NativeProcessProtocol *process) = 0;
  };

  /// Register a native delegate.
  ///
  /// Clients can register nofication callbacks by passing in a
  /// NativeDelegate impl and passing it into this function.
  ///
  /// Note: it is required that the lifetime of the
  /// native_delegate outlive the NativeProcessProtocol.
  ///
  /// \param[in] native_delegate
  ///     A NativeDelegate impl to be called when certain events
  ///     happen within the NativeProcessProtocol or related threads.
  ///
  /// \return
  ///     true if the delegate was registered successfully;
  ///     false if the delegate was already registered.
  ///
  /// \see NativeProcessProtocol::NativeDelegate.
  bool RegisterNativeDelegate(NativeDelegate &native_delegate);

  /// Unregister a native delegate previously registered.
  ///
  /// \param[in] native_delegate
  ///     A NativeDelegate impl previously registered with this process.
  ///
  /// \return Returns \b true if the NativeDelegate was
  /// successfully removed from the process, \b false otherwise.
  ///
  /// \see NativeProcessProtocol::NativeDelegate
  bool UnregisterNativeDelegate(NativeDelegate &native_delegate);

  virtual Status GetLoadedModuleFileSpec(const char *module_path,
                                         FileSpec &file_spec) = 0;

  virtual Status GetFileLoadAddress(const llvm::StringRef &file_name,
                                    lldb::addr_t &load_addr) = 0;

  class Factory {
  public:
    virtual ~Factory();
    /// Launch a process for debugging.
    ///
    /// \param[in] launch_info
    ///     Information required to launch the process.
    ///
    /// \param[in] native_delegate
    ///     The delegate that will receive messages regarding the
    ///     inferior.  Must outlive the NativeProcessProtocol
    ///     instance.
    ///
    /// \param[in] mainloop
    ///     The mainloop instance with which the process can register
    ///     callbacks. Must outlive the NativeProcessProtocol
    ///     instance.
    ///
    /// \return
    ///     A NativeProcessProtocol shared pointer if the operation succeeded or
    ///     an error object if it failed.
    virtual llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
    Launch(ProcessLaunchInfo &launch_info, NativeDelegate &native_delegate,
           MainLoop &mainloop) const = 0;

    /// Attach to an existing process.
    ///
    /// \param[in] pid
    ///     pid of the process locatable
    ///
    /// \param[in] native_delegate
    ///     The delegate that will receive messages regarding the
    ///     inferior.  Must outlive the NativeProcessProtocol
    ///     instance.
    ///
    /// \param[in] mainloop
    ///     The mainloop instance with which the process can register
    ///     callbacks. Must outlive the NativeProcessProtocol
    ///     instance.
    ///
    /// \return
    ///     A NativeProcessProtocol shared pointer if the operation succeeded or
    ///     an error object if it failed.
    virtual llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
    Attach(lldb::pid_t pid, NativeDelegate &native_delegate,
           MainLoop &mainloop) const = 0;
  };

  /// StartTracing API for starting a tracing instance with the
  /// TraceOptions on a specific thread or process.
  ///
  /// \param[in] config
  ///     The configuration to use when starting tracing.
  ///
  /// \param[out] error
  ///     Status indicates what went wrong.
  ///
  /// \return
  ///     The API returns a user_id which can be used to get trace
  ///     data, trace configuration or stopping the trace instance.
  ///     The user_id is a key to identify and operate with a tracing
  ///     instance. It may refer to the complete process or a single
  ///     thread.
  virtual lldb::user_id_t StartTrace(const TraceOptions &config,
                                     Status &error) {
    error.SetErrorString("Not implemented");
    return LLDB_INVALID_UID;
  }

  /// StopTracing API as the name suggests stops a tracing instance.
  ///
  /// \param[in] traceid
  ///     The user id of the trace intended to be stopped. Now a
  ///     user_id may map to multiple threads in which case this API
  ///     could be used to stop the tracing for a specific thread by
  ///     supplying its thread id.
  ///
  /// \param[in] thread
  ///     Thread is needed when the complete process is being traced
  ///     and the user wishes to stop tracing on a particular thread.
  ///
  /// \return
  ///     Status indicating what went wrong.
  virtual Status StopTrace(lldb::user_id_t traceid,
                           lldb::tid_t thread = LLDB_INVALID_THREAD_ID) {
    return Status("Not implemented");
  }

  /// This API provides the trace data collected in the form of raw
  /// data.
  ///
  /// \param[in] traceid thread
  ///     The traceid and thread provide the context for the trace
  ///     instance.
  ///
  /// \param[in] buffer
  ///     The buffer provides the destination buffer where the trace
  ///     data would be read to. The buffer should be truncated to the
  ///     filled length by this function.
  ///
  /// \param[in] offset
  ///     There is possibility to read partially the trace data from
  ///     a specified offset where in such cases the buffer provided
  ///     may be smaller than the internal trace collection container.
  ///
  /// \return
  ///     The size of the data actually read.
  virtual Status GetData(lldb::user_id_t traceid, lldb::tid_t thread,
                         llvm::MutableArrayRef<uint8_t> &buffer,
                         size_t offset = 0) {
    return Status("Not implemented");
  }

  /// Similar API as above except it aims to provide any extra data
  /// useful for decoding the actual trace data.
  virtual Status GetMetaData(lldb::user_id_t traceid, lldb::tid_t thread,
                             llvm::MutableArrayRef<uint8_t> &buffer,
                             size_t offset = 0) {
    return Status("Not implemented");
  }

  /// API to query the TraceOptions for a given user id
  ///
  /// \param[in] traceid
  ///     The user id of the tracing instance.
  ///
  /// \param[out] config
  ///     The configuration being used for tracing.
  ///
  /// \return A status indicating what went wrong.
  virtual Status GetTraceConfig(lldb::user_id_t traceid, TraceOptions &config) {
    return Status("Not implemented");
  }

  /// \copydoc Process::GetSupportedTraceType()
  virtual llvm::Expected<TraceTypeInfo> GetSupportedTraceType() {
    return llvm::make_error<UnimplementedError>();
  }

protected:
  struct SoftwareBreakpoint {
    uint32_t ref_count;
    llvm::SmallVector<uint8_t, 4> saved_opcodes;
    llvm::ArrayRef<uint8_t> breakpoint_opcodes;
  };

  std::unordered_map<lldb::addr_t, SoftwareBreakpoint> m_software_breakpoints;
  lldb::pid_t m_pid;

  std::vector<std::unique_ptr<NativeThreadProtocol>> m_threads;
  lldb::tid_t m_current_thread_id = LLDB_INVALID_THREAD_ID;
  mutable std::recursive_mutex m_threads_mutex;

  lldb::StateType m_state = lldb::eStateInvalid;
  mutable std::recursive_mutex m_state_mutex;

  llvm::Optional<WaitStatus> m_exit_status;

  std::recursive_mutex m_delegates_mutex;
  std::vector<NativeDelegate *> m_delegates;
  NativeWatchpointList m_watchpoint_list;
  HardwareBreakpointMap m_hw_breakpoints_map;
  int m_terminal_fd;
  uint32_t m_stop_id = 0;

  // Set of signal numbers that LLDB directly injects back to inferior without
  // stopping it.
  llvm::DenseSet<int> m_signals_to_ignore;

  // lldb_private::Host calls should be used to launch a process for debugging,
  // and then the process should be attached to. When attaching to a process
  // lldb_private::Host calls should be used to locate the process to attach
  // to, and then this function should be called.
  NativeProcessProtocol(lldb::pid_t pid, int terminal_fd,
                        NativeDelegate &delegate);

  void SetID(lldb::pid_t pid) { m_pid = pid; }

  // interface for state handling
  void SetState(lldb::StateType state, bool notify_delegates = true);

  // Derived classes need not implement this.  It can be used as a hook to
  // clear internal caches that should be invalidated when stop ids change.
  //
  // Note this function is called with the state mutex obtained by the caller.
  virtual void DoStopIDBumped(uint32_t newBumpId);

  // interface for software breakpoints

  Status SetSoftwareBreakpoint(lldb::addr_t addr, uint32_t size_hint);
  Status RemoveSoftwareBreakpoint(lldb::addr_t addr);

  virtual llvm::Expected<llvm::ArrayRef<uint8_t>>
  GetSoftwareBreakpointTrapOpcode(size_t size_hint);

  /// Return the offset of the PC relative to the software breakpoint that was hit. If an
  /// architecture (e.g. arm) reports breakpoint hits before incrementing the PC, this offset
  /// will be 0. If an architecture (e.g. intel) reports breakpoints hits after incrementing the
  /// PC, this offset will be the size of the breakpoint opcode.
  virtual size_t GetSoftwareBreakpointPCOffset();

  // Adjust the thread's PC after hitting a software breakpoint. On
  // architectures where the PC points after the breakpoint instruction, this
  // resets it to point to the breakpoint itself.
  void FixupBreakpointPCAsNeeded(NativeThreadProtocol &thread);

  /// Notify the delegate that an exec occurred.
  ///
  /// Provide a mechanism for a delegate to clear out any exec-
  /// sensitive data.
  void NotifyDidExec();

  NativeThreadProtocol *GetThreadByIDUnlocked(lldb::tid_t tid);

private:
  void SynchronouslyNotifyProcessStateChanged(lldb::StateType state);
  llvm::Expected<SoftwareBreakpoint>
  EnableSoftwareBreakpoint(lldb::addr_t addr, uint32_t size_hint);
};
} // namespace lldb_private

#endif // LLDB_HOST_COMMON_NATIVEPROCESSPROTOCOL_H
