//===-- Host.h --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Host_h_
#define liblldb_Host_h_
#if defined(__cplusplus)

#include <stdarg.h>

#include <map>
#include <string>

#include "lldb/lldb-private.h"
#include "lldb/Core/StringList.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class Host Host.h "lldb/Host/Host.h"
/// @brief A class that provides host computer information.
///
/// Host is a class that answers information about the host operating
/// system.
//----------------------------------------------------------------------
class Host
{
public:
    typedef bool (*MonitorChildProcessCallback) (void *callback_baton,
                                                 lldb::pid_t pid,
                                                 bool exited,
                                                 int signal,    // Zero for no signal
                                                 int status);   // Exit value of process if signal is zero

    //------------------------------------------------------------------
    /// Start monitoring a child process.
    ///
    /// Allows easy monitoring of child processes. \a callback will be
    /// called when the child process exits or if it gets a signal. The
    /// callback will only be called with signals if \a monitor_signals
    /// is \b true. \a callback will usually be called from another
    /// thread so the callback function must be thread safe.
    ///
    /// When the callback gets called, the return value indicates if
    /// minotoring should stop. If \b true is returned from \a callback
    /// the information will be removed. If \b false is returned then
    /// monitoring will continue. If the child process exits, the
    /// monitoring will automatically stop after the callback returned
    /// ragardless of the callback return value.
    ///
    /// @param[in] callback
    ///     A function callback to call when a child receives a signal
    ///     (if \a monitor_signals is true) or a child exits.
    ///
    /// @param[in] callback_baton
    ///     A void * of user data that will be pass back when
    ///     \a callback is called.
    ///
    /// @param[in] pid
    ///     The process ID of a child process to monitor, -1 for all
    ///     processes.
    ///
    /// @param[in] monitor_signals
    ///     If \b true the callback will get called when the child
    ///     process gets a signal. If \b false, the callback will only
    ///     get called if the child process exits.
    ///
    /// @return
    ///     A thread handle that can be used to cancel the thread that
    ///     was spawned to monitor \a pid.
    ///
    /// @see static void Host::StopMonitoringChildProcess (uint32_t)
    //------------------------------------------------------------------
    static lldb::thread_t
    StartMonitoringChildProcess (MonitorChildProcessCallback callback,
                                 void *callback_baton,
                                 lldb::pid_t pid,
                                 bool monitor_signals);

    //------------------------------------------------------------------
    /// Get the host page size.
    ///
    /// @return
    ///     The size in bytes of a VM page on the host system.
    //------------------------------------------------------------------
    static size_t
    GetPageSize();

    //------------------------------------------------------------------
    /// Returns the endianness of the host system.
    ///
    /// @return
    ///     Returns the endianness of the host system as a lldb::ByteOrder
    ///     enumeration.
    //------------------------------------------------------------------
    static lldb::ByteOrder
    GetByteOrder ();

    //------------------------------------------------------------------
    /// Returns the number of CPUs on this current host.
    ///
    /// @return
    ///     Number of CPUs on this current host, or zero if the number
    ///     of CPUs can't be determined on this host.
    //------------------------------------------------------------------
    static uint32_t
    GetNumberCPUS ();

    static bool
    GetOSVersion (uint32_t &major, 
                  uint32_t &minor, 
                  uint32_t &update);

    static bool
    GetOSBuildString (std::string &s);
    
    static bool
    GetOSKernelDescription (std::string &s);

    static bool
    GetHostname (std::string &s);

    static const char *
    GetUserName (uint32_t uid, std::string &user_name);
    
    static const char *
    GetGroupName (uint32_t gid, std::string &group_name);
    
    static uint32_t
    GetUserID ();
    
    static uint32_t
    GetGroupID ();

    static uint32_t
    GetEffectiveUserID ();

    static uint32_t
    GetEffectiveGroupID ();


    enum SystemLogType
    {
        eSystemLogWarning,
        eSystemLogError
    };

    static void
    SystemLog (SystemLogType type, const char *format, ...) __attribute__ ((format (printf, 2, 3)));

    static void
    SystemLog (SystemLogType type, const char *format, va_list args);

    //------------------------------------------------------------------
    /// Gets the host architecture.
    ///
    /// @return
    ///     A const architecture object that represents the host
    ///     architecture.
    //------------------------------------------------------------------
    enum SystemDefaultArchitecture
    {
        eSystemDefaultArchitecture,     // The overall default architecture that applications will run on this host
        eSystemDefaultArchitecture32,   // If this host supports 32 bit programs, return the default 32 bit arch
        eSystemDefaultArchitecture64    // If this host supports 64 bit programs, return the default 64 bit arch
    };

    static const ArchSpec &
    GetArchitecture (SystemDefaultArchitecture arch_kind = eSystemDefaultArchitecture);

    //------------------------------------------------------------------
    /// Gets the host vendor string.
    ///
    /// @return
    ///     A const string object containing the host vendor name.
    //------------------------------------------------------------------
    static const ConstString &
    GetVendorString ();

    //------------------------------------------------------------------
    /// Gets the host Operating System (OS) string.
    ///
    /// @return
    ///     A const string object containing the host OS name.
    //------------------------------------------------------------------
    static const ConstString &
    GetOSString ();

    //------------------------------------------------------------------
    /// Gets the host target triple as a const string.
    ///
    /// @return
    ///     A const string object containing the host target triple.
    //------------------------------------------------------------------
    static const ConstString &
    GetTargetTriple ();

    //------------------------------------------------------------------
    /// Get the process ID for the calling process.
    ///
    /// @return
    ///     The process ID for the current process.
    //------------------------------------------------------------------
    static lldb::pid_t
    GetCurrentProcessID ();

    //------------------------------------------------------------------
    /// Get the thread ID for the calling thread in the current process.
    ///
    /// @return
    ///     The thread ID for the calling thread in the current process.
    //------------------------------------------------------------------
    static lldb::tid_t
    GetCurrentThreadID ();

    //------------------------------------------------------------------
    /// Get the thread token (the one returned by ThreadCreate when the thread was created) for the
    /// calling thread in the current process.
    ///
    /// @return
    ///     The thread token for the calling thread in the current process.
    //------------------------------------------------------------------
    static lldb::thread_t
    GetCurrentThread ();

    static const char *
    GetSignalAsCString (int signo);

    static void
    WillTerminate ();
    //------------------------------------------------------------------
    /// Host specific thread created function call.
    ///
    /// This function call lets the current host OS do any thread
    /// specific initialization that it needs, including naming the
    /// thread. No cleanup routine is exptected to be called
    ///
    /// @param[in] name
    ///     The current thread's name in the current process.
    //------------------------------------------------------------------
    static void
    ThreadCreated (const char *name);

    static lldb::thread_t
    ThreadCreate (const char *name,
                  lldb::thread_func_t function,
                  lldb::thread_arg_t thread_arg,
                  Error *err);

    static bool
    ThreadCancel (lldb::thread_t thread,
                  Error *error);

    static bool
    ThreadDetach (lldb::thread_t thread,
                  Error *error);
    static bool
    ThreadJoin (lldb::thread_t thread,
                lldb::thread_result_t *thread_result_ptr,
                Error *error);

    //------------------------------------------------------------------
    /// Gets the name of a thread in a process.
    ///
    /// This function will name a thread in a process using it's own
    /// thread name pool, and also will attempt to set a thread name
    /// using any supported host OS APIs.
    ///
    /// @param[in] pid
    ///     The process ID in which we are trying to get the name of
    ///     a thread.
    ///
    /// @param[in] tid
    ///     The thread ID for which we are trying retrieve the name of.
    ///
    /// @return
    ///     A std::string containing the thread name.
    //------------------------------------------------------------------
    static std::string
    GetThreadName (lldb::pid_t pid, lldb::tid_t tid);

    //------------------------------------------------------------------
    /// Sets the name of a thread in the current process.
    ///
    /// @param[in] pid
    ///     The process ID in which we are trying to name a thread.
    ///
    /// @param[in] tid
    ///     The thread ID which we are trying to name.
    ///
    /// @param[in] name
    ///     The current thread's name in the current process to \a name.
    ///
    /// @return
    ///     \b true if the thread name was able to be set, \b false
    ///     otherwise.
    //------------------------------------------------------------------
    static bool
    SetThreadName (lldb::pid_t pid, lldb::tid_t tid, const char *name);

    //------------------------------------------------------------------
    /// Gets the FileSpec of the current process (the process that
    /// that is running the LLDB code).
    ///
    /// @return
    ///     \b A file spec with the program name.
    //------------------------------------------------------------------
    static FileSpec
    GetProgramFileSpec ();

    //------------------------------------------------------------------
    /// Given an address in the current process (the process that
    /// is running the LLDB code), return the name of the module that
    /// it comes from. This can be useful when you need to know the
    /// path to the shared library that your code is running in for
    /// loading resources that are relative to your binary.
    ///
    /// @param[in] host_addr
    ///     The pointer to some code in the current process.
    ///
    /// @return
    ///     \b A file spec with the module that contains \a host_addr,
    ///     which may be invalid if \a host_addr doesn't fall into
    ///     any valid module address range.
    //------------------------------------------------------------------
    static FileSpec
    GetModuleFileSpecForHostAddress (const void *host_addr);


    
    //------------------------------------------------------------------
    /// If you have an executable that is in a bundle and want to get
    /// back to the bundle directory from the path itself, this 
    /// function will change a path to a file within a bundle to the
    /// bundle directory itself.
    ///
    /// @param[in] file
    ///     A file spec that might point to a file in a bundle. 
    ///
    /// @param[out] bundle_directory
    ///     An object will be filled in with the bundle directory for
    ///     the bundle when \b true is returned. Otherwise \a file is 
    ///     left untouched and \b false is returned.
    ///
    /// @return
    ///     \b true if \a file was resolved in \a bundle_directory,
    ///     \b false otherwise.
    //------------------------------------------------------------------
    static bool
    GetBundleDirectory (const FileSpec &file, FileSpec &bundle_directory);

    //------------------------------------------------------------------
    /// When executable files may live within a directory, where the 
    /// directory represents an executable bundle (like the MacOSX 
    /// app bundles), the locate the executable within the containing
    /// bundle.
    ///
    /// @param[in,out] file
    ///     A file spec that currently points to the bundle that will
    ///     be filled in with the executable path within the bundle
    ///     if \b true is returned. Otherwise \a file is left untouched.
    ///
    /// @return
    ///     \b true if \a file was resolved, \b false if this function
    ///     was not able to resolve the path.
    //------------------------------------------------------------------
    static bool
    ResolveExecutableInBundle (FileSpec &file);

    //------------------------------------------------------------------
    /// Find a resource files that are related to LLDB.
    ///
    /// Operating systems have different ways of storing shared 
    /// libraries and related resources. This function abstracts the
    /// access to these paths.
    ///
    /// @param[in] path_type
    ///     The type of LLDB resource path you are looking for. If the
    ///     enumeration ends with "Dir", then only the \a file_spec's 
    ///     directory member gets filled in.
    ///
    /// @param[in] file_spec
    ///     A file spec that gets filled in with the appriopriate path.
    ///
    /// @return
    ///     \b true if \a resource_path was resolved, \a false otherwise.
    //------------------------------------------------------------------
    static bool
    GetLLDBPath (PathType path_type,
                 FileSpec &file_spec);

    //------------------------------------------------------------------
    /// Set a string that can be displayed if host application crashes.
    ///
    /// Some operating systems have the ability to print a description
    /// for shared libraries when a program crashes. If the host OS
    /// supports such a mechanism, it should be implemented to help
    /// with crash triage.
    ///
    /// @param[in] format
    ///     A printf format that will be used to form a new crash
    ///     description string.
    //------------------------------------------------------------------
    static void
    SetCrashDescriptionWithFormat (const char *format, ...)  __attribute__ ((format (printf, 1, 2)));

    static void
    SetCrashDescription (const char *description);

    static uint32_t
    FindProcesses (const ProcessInstanceInfoMatch &match_info,
                   ProcessInstanceInfoList &proc_infos);

    typedef std::map<lldb::pid_t, bool> TidMap;
    typedef std::pair<lldb::pid_t, bool> TidPair;
    static bool
    FindProcessThreads (const lldb::pid_t pid, TidMap &tids_to_attach);

    static bool
    GetProcessInfo (lldb::pid_t pid, ProcessInstanceInfo &proc_info);
    
    static lldb::pid_t
    LaunchApplication (const FileSpec &app_file_spec);

    static Error
    LaunchProcess (ProcessLaunchInfo &launch_info);

    static Error
    RunShellCommand (const char *command,           // Shouldn't be NULL
                     const char *working_dir,       // Pass NULL to use the current working directory
                     int *status_ptr,               // Pass NULL if you don't want the process exit status
                     int *signo_ptr,                // Pass NULL if you don't want the signal that caused the process to exit
                     std::string *command_output,   // Pass NULL if you don't want the command output
                     uint32_t timeout_sec,
                     const char *shell = "/bin/bash");
    
    static lldb::DataBufferSP
    GetAuxvData (lldb_private::Process *process);

    static lldb::TargetSP
    GetDummyTarget (Debugger &debugger);
    
    static bool
    OpenFileInExternalEditor (const FileSpec &file_spec, 
                              uint32_t line_no);

    static void
    Backtrace (Stream &strm, uint32_t max_frames);
    
    static size_t
    GetEnvironment (StringList &env);

    enum DynamicLibraryOpenOptions 
    {
        eDynamicLibraryOpenOptionLazy           = (1u << 0),  // Lazily resolve symbols in this dynamic library
        eDynamicLibraryOpenOptionLocal          = (1u << 1),  // Only open a shared library with local access (hide it from the global symbol namespace)
        eDynamicLibraryOpenOptionLimitGetSymbol = (1u << 2)   // DynamicLibraryGetSymbol calls on this handle will only return matches from this shared library
    };
    static void *
    DynamicLibraryOpen (const FileSpec &file_spec, 
                        uint32_t options,
                        Error &error);

    static Error
    DynamicLibraryClose (void *dynamic_library_handle);

    static void *
    DynamicLibraryGetSymbol (void *dynamic_library_handle, 
                             const char *symbol_name, 
                             Error &error);
};

} // namespace lldb_private

#endif  // #if defined(__cplusplus)
#endif  // liblldb_Host_h_
