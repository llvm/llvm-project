//===-- NativeProcessProtocol.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_NativeProcessProtocol_h_
#define liblldb_NativeProcessProtocol_h_

#include <mutex>
#include <vector>

#include "lldb/lldb-private-forward.h"
#include "lldb/lldb-types.h"
#include "lldb/Core/Error.h"
#include "lldb/Host/MainLoop.h"
#include "llvm/ADT/StringRef.h"

#include "NativeBreakpointList.h"
#include "NativeWatchpointList.h"

namespace lldb_private
{
    class MemoryRegionInfo;
    class ResumeActionList;

    //------------------------------------------------------------------
    // NativeProcessProtocol
    //------------------------------------------------------------------
    class NativeProcessProtocol :
        public std::enable_shared_from_this<NativeProcessProtocol>
    {
        friend class SoftwareBreakpoint;

    public:

        virtual ~NativeProcessProtocol ()
        {
        }

        virtual Error
        Resume (const ResumeActionList &resume_actions) = 0;

        virtual Error
        Halt () = 0;

        virtual Error
        Detach () = 0;

        //------------------------------------------------------------------
        /// Sends a process a UNIX signal \a signal.
        ///
        /// @return
        ///     Returns an error object.
        //------------------------------------------------------------------
        virtual Error
        Signal (int signo) = 0;

        //------------------------------------------------------------------
        /// Tells a process to interrupt all operations as if by a Ctrl-C.
        ///
        /// The default implementation will send a local host's equivalent of
        /// a SIGSTOP to the process via the NativeProcessProtocol::Signal()
        /// operation.
        ///
        /// @return
        ///     Returns an error object.
        //------------------------------------------------------------------
        virtual Error
        Interrupt ();

        virtual Error
        Kill () = 0;

        //----------------------------------------------------------------------
        // Memory and memory region functions
        //----------------------------------------------------------------------

        virtual Error
        GetMemoryRegionInfo (lldb::addr_t load_addr, MemoryRegionInfo &range_info);

        virtual Error
        ReadMemory(lldb::addr_t addr, void *buf, size_t size, size_t &bytes_read) = 0;

        virtual Error
        ReadMemoryWithoutTrap(lldb::addr_t addr, void *buf, size_t size, size_t &bytes_read) = 0;

        virtual Error
        WriteMemory(lldb::addr_t addr, const void *buf, size_t size, size_t &bytes_written) = 0;

        virtual Error
        AllocateMemory(size_t size, uint32_t permissions, lldb::addr_t &addr) = 0;

        virtual Error
        DeallocateMemory (lldb::addr_t addr) = 0;

        virtual lldb::addr_t
        GetSharedLibraryInfoAddress () = 0;

        virtual bool
        IsAlive () const;

        virtual size_t
        UpdateThreads () = 0;

        virtual bool
        GetArchitecture (ArchSpec &arch) const = 0;

        //----------------------------------------------------------------------
        // Breakpoint functions
        //----------------------------------------------------------------------
        virtual Error
        SetBreakpoint (lldb::addr_t addr, uint32_t size, bool hardware) = 0;

        virtual Error
        RemoveBreakpoint (lldb::addr_t addr);

        virtual Error
        EnableBreakpoint (lldb::addr_t addr);

        virtual Error
        DisableBreakpoint (lldb::addr_t addr);

        //----------------------------------------------------------------------
        // Watchpoint functions
        //----------------------------------------------------------------------
        virtual const NativeWatchpointList::WatchpointMap&
        GetWatchpointMap () const;

        virtual uint32_t
        GetMaxWatchpoints () const;

        virtual Error
        SetWatchpoint (lldb::addr_t addr, size_t size, uint32_t watch_flags, bool hardware);

        virtual Error
        RemoveWatchpoint (lldb::addr_t addr);

        //----------------------------------------------------------------------
        // Accessors
        //----------------------------------------------------------------------
        lldb::pid_t
        GetID() const
        {
            return m_pid;
        }

        lldb::StateType
        GetState () const;

        bool
        IsRunning () const
        {
            return m_state == lldb::eStateRunning || IsStepping();
        }

        bool
        IsStepping () const
        {
            return m_state == lldb::eStateStepping;
        }

        bool
        CanResume () const
        {
            return m_state == lldb::eStateStopped;
        }

        bool
        GetByteOrder (lldb::ByteOrder &byte_order) const;

        //----------------------------------------------------------------------
        // Exit Status
        //----------------------------------------------------------------------
        virtual bool
        GetExitStatus (lldb_private::ExitType *exit_type, int *status, std::string &exit_description);

        virtual bool
        SetExitStatus (lldb_private::ExitType exit_type, int status, const char *exit_description, bool bNotifyStateChange);

        //----------------------------------------------------------------------
        // Access to threads
        //----------------------------------------------------------------------
        NativeThreadProtocolSP
        GetThreadAtIndex (uint32_t idx);

        NativeThreadProtocolSP
        GetThreadByID (lldb::tid_t tid);

        void
        SetCurrentThreadID (lldb::tid_t tid)
        {
            m_current_thread_id = tid;
        }

        lldb::tid_t
        GetCurrentThreadID ()
        {
            return m_current_thread_id;
        }

        NativeThreadProtocolSP
        GetCurrentThread ()
        {
            return GetThreadByID (m_current_thread_id);
        }

        //----------------------------------------------------------------------
        // Access to inferior stdio
        //----------------------------------------------------------------------
        virtual
        int GetTerminalFileDescriptor ()
        {
            return m_terminal_fd;
        }

        //----------------------------------------------------------------------
        // Stop id interface
        //----------------------------------------------------------------------

        uint32_t
        GetStopID () const;

        // ---------------------------------------------------------------------
        // Callbacks for low-level process state changes
        // ---------------------------------------------------------------------
        class NativeDelegate
        {
        public:
            virtual
            ~NativeDelegate () {}

            virtual void
            InitializeDelegate (NativeProcessProtocol *process) = 0;

            virtual void
            ProcessStateChanged (NativeProcessProtocol *process, lldb::StateType state) = 0;

            virtual void
            DidExec (NativeProcessProtocol *process) = 0;
        };

        //------------------------------------------------------------------
        /// Register a native delegate.
        ///
        /// Clients can register nofication callbacks by passing in a
        /// NativeDelegate impl and passing it into this function.
        ///
        /// Note: it is required that the lifetime of the
        /// native_delegate outlive the NativeProcessProtocol.
        ///
        /// @param[in] native_delegate
        ///     A NativeDelegate impl to be called when certain events
        ///     happen within the NativeProcessProtocol or related threads.
        ///
        /// @return
        ///     true if the delegate was registered successfully;
        ///     false if the delegate was already registered.
        ///
        /// @see NativeProcessProtocol::NativeDelegate.
        //------------------------------------------------------------------
        bool
        RegisterNativeDelegate (NativeDelegate &native_delegate);

        //------------------------------------------------------------------
        /// Unregister a native delegate previously registered.
        ///
        /// @param[in] native_delegate
        ///     A NativeDelegate impl previously registered with this process.
        ///
        /// @return Returns \b true if the NativeDelegate was
        /// successfully removed from the process, \b false otherwise.
        ///
        /// @see NativeProcessProtocol::NativeDelegate
        //------------------------------------------------------------------
        bool
        UnregisterNativeDelegate (NativeDelegate &native_delegate);

        virtual Error
        GetLoadedModuleFileSpec(const char* module_path, FileSpec& file_spec) = 0;

        virtual Error
        GetFileLoadAddress(const llvm::StringRef& file_name, lldb::addr_t& load_addr) = 0;

        //------------------------------------------------------------------
        /// Launch a process for debugging. This method will create an concrete
        /// instance of NativeProcessProtocol, based on the host platform.
        /// (e.g. NativeProcessLinux on linux, etc.)
        ///
        /// @param[in] launch_info
        ///     Information required to launch the process.
        ///
        /// @param[in] native_delegate
        ///     The delegate that will receive messages regarding the
        ///     inferior.  Must outlive the NativeProcessProtocol
        ///     instance.
        ///
        /// @param[in] mainloop
        ///     The mainloop instance with which the process can register
        ///     callbacks. Must outlive the NativeProcessProtocol
        ///     instance.
        ///
        /// @param[out] process_sp
        ///     On successful return from the method, this parameter
        ///     contains the shared pointer to the
        ///     NativeProcessProtocol that can be used to manipulate
        ///     the native process.
        ///
        /// @return
        ///     An error object indicating if the operation succeeded,
        ///     and if not, what error occurred.
        //------------------------------------------------------------------
        static Error
        Launch (ProcessLaunchInfo &launch_info,
                NativeDelegate &native_delegate,
                MainLoop &mainloop,
                NativeProcessProtocolSP &process_sp);

        //------------------------------------------------------------------
        /// Attach to an existing process. This method will create an concrete
        /// instance of NativeProcessProtocol, based on the host platform.
        /// (e.g. NativeProcessLinux on linux, etc.)
        ///
        /// @param[in] pid
        ///     pid of the process locatable
        ///
        /// @param[in] native_delegate
        ///     The delegate that will receive messages regarding the
        ///     inferior.  Must outlive the NativeProcessProtocol
        ///     instance.
        ///
        /// @param[in] mainloop
        ///     The mainloop instance with which the process can register
        ///     callbacks. Must outlive the NativeProcessProtocol
        ///     instance.
        ///
        /// @param[out] process_sp
        ///     On successful return from the method, this parameter
        ///     contains the shared pointer to the
        ///     NativeProcessProtocol that can be used to manipulate
        ///     the native process.
        ///
        /// @return
        ///     An error object indicating if the operation succeeded,
        ///     and if not, what error occurred.
        //------------------------------------------------------------------
        static Error
        Attach (lldb::pid_t pid,
                NativeDelegate &native_delegate,
                MainLoop &mainloop,
                NativeProcessProtocolSP &process_sp);

    protected:
        lldb::pid_t m_pid;

        std::vector<NativeThreadProtocolSP> m_threads;
        lldb::tid_t m_current_thread_id;
        mutable std::recursive_mutex m_threads_mutex;

        lldb::StateType m_state;
        mutable std::recursive_mutex m_state_mutex;

        lldb_private::ExitType m_exit_type;
        int m_exit_status;
        std::string m_exit_description;
        std::recursive_mutex m_delegates_mutex;
        std::vector<NativeDelegate*> m_delegates;
        NativeBreakpointList m_breakpoint_list;
        NativeWatchpointList m_watchpoint_list;
        int m_terminal_fd;
        uint32_t m_stop_id;

        // lldb_private::Host calls should be used to launch a process for debugging, and
        // then the process should be attached to. When attaching to a process
        // lldb_private::Host calls should be used to locate the process to attach to,
        // and then this function should be called.
        NativeProcessProtocol (lldb::pid_t pid);

        // -----------------------------------------------------------
        // Internal interface for state handling
        // -----------------------------------------------------------
        void
        SetState (lldb::StateType state, bool notify_delegates = true);

        // Derived classes need not implement this.  It can be used as a
        // hook to clear internal caches that should be invalidated when
        // stop ids change.
        //
        // Note this function is called with the state mutex obtained
        // by the caller.
        virtual void
        DoStopIDBumped (uint32_t newBumpId);

        // -----------------------------------------------------------
        // Internal interface for software breakpoints
        // -----------------------------------------------------------
        Error
        SetSoftwareBreakpoint (lldb::addr_t addr, uint32_t size_hint);

        virtual Error
        GetSoftwareBreakpointTrapOpcode (size_t trap_opcode_size_hint, size_t &actual_opcode_size, const uint8_t *&trap_opcode_bytes) = 0;

        // -----------------------------------------------------------
        /// Notify the delegate that an exec occurred.
        ///
        /// Provide a mechanism for a delegate to clear out any exec-
        /// sensitive data.
        // -----------------------------------------------------------
        void
        NotifyDidExec ();

        NativeThreadProtocolSP
        GetThreadByIDUnlocked (lldb::tid_t tid);

        // -----------------------------------------------------------
        // Static helper methods for derived classes.
        // -----------------------------------------------------------
        static Error
        ResolveProcessArchitecture(lldb::pid_t pid, ArchSpec &arch);

    private:

        void
        SynchronouslyNotifyProcessStateChanged (lldb::StateType state);
    };
}

#endif // #ifndef liblldb_NativeProcessProtocol_h_
