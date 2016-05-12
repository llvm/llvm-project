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
#include "lldb/lldb-private-forward.h"
#include "lldb/Core/StringList.h"
#include "lldb/Host/File.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/HostThread.h"

namespace lldb_private {

class FileAction;
class ProcessLaunchInfo;

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
    typedef std::function<bool(lldb::pid_t pid, bool exited,
                               int signal,  // Zero for no signal
                               int status)> // Exit value of process if signal is zero
        MonitorChildProcessCallback;

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
    /// monitoring should stop. If \b true is returned from \a callback
    /// the information will be removed. If \b false is returned then
    /// monitoring will continue. If the child process exits, the
    /// monitoring will automatically stop after the callback returned
    /// regardless of the callback return value.
    ///
    /// @param[in] callback
    ///     A function callback to call when a child receives a signal
    ///     (if \a monitor_signals is true) or a child exits.
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
    static HostThread
    StartMonitoringChildProcess(const MonitorChildProcessCallback &callback, lldb::pid_t pid, bool monitor_signals);

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
    /// Get the process ID for the calling process.
    ///
    /// @return
    ///     The process ID for the current process.
    //------------------------------------------------------------------
    static lldb::pid_t
    GetCurrentProcessID ();

    static void
    Kill(lldb::pid_t pid, int signo);

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

    typedef void (*ThreadLocalStorageCleanupCallback) (void *p);

    static lldb::thread_key_t
    ThreadLocalStorageCreate(ThreadLocalStorageCleanupCallback callback);

    static void*
    ThreadLocalStorageGet(lldb::thread_key_t key);

    static void
    ThreadLocalStorageSet(lldb::thread_key_t key, void *value);


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
    /// app bundles), then locate the executable within the containing
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

#if defined (__APPLE__) || defined (__linux__) || defined (__FreeBSD__) || defined (__GLIBC__) || defined (__NetBSD__)
#if !defined(__ANDROID__) && !defined(__ANDROID_NDK__)

    static short GetPosixspawnFlags(const ProcessLaunchInfo &launch_info);

    static Error LaunchProcessPosixSpawn(const char *exe_path, const ProcessLaunchInfo &launch_info, lldb::pid_t &pid);

    static bool AddPosixSpawnFileAction(void *file_actions, const FileAction *info, Log *log, Error &error);

#endif // !defined(__ANDROID__) && !defined(__ANDROID_NDK__)
#endif // defined (__APPLE__) || defined (__linux__) || defined (__FreeBSD__) || defined (__GLIBC__) || defined(__NetBSD__)

    static const lldb::UnixSignalsSP &
    GetUnixSignals();

    static Error
    LaunchProcess (ProcessLaunchInfo &launch_info);

    //------------------------------------------------------------------
    /// Perform expansion of the command-line for this launch info
    /// This can potentially involve wildcard expansion
    //  environment variable replacement, and whatever other
    //  argument magic the platform defines as part of its typical
    //  user experience
    //------------------------------------------------------------------
    static Error
    ShellExpandArguments (ProcessLaunchInfo &launch_info);
    
    static Error
    RunShellCommand(const char *command,           // Shouldn't be NULL
                    const FileSpec &working_dir,   // Pass empty FileSpec to use the current working directory
                    int *status_ptr,               // Pass NULL if you don't want the process exit status
                    int *signo_ptr,                // Pass NULL if you don't want the signal that caused the process to exit
                    std::string *command_output,   // Pass NULL if you don't want the command output
                    uint32_t timeout_sec,
                    bool run_in_default_shell = true);

    static Error
    RunShellCommand(const Args& args,
                    const FileSpec &working_dir,   // Pass empty FileSpec to use the current working directory
                    int *status_ptr,               // Pass NULL if you don't want the process exit status
                    int *signo_ptr,                // Pass NULL if you don't want the signal that caused the process to exit
                    std::string *command_output,   // Pass NULL if you don't want the command output
                    uint32_t timeout_sec,
                    bool run_in_default_shell = true);
    
    static lldb::DataBufferSP
    GetAuxvData (lldb_private::Process *process);

    static lldb::DataBufferSP
    GetAuxvData (lldb::pid_t pid);

    static bool
    OpenFileInExternalEditor (const FileSpec &file_spec, 
                              uint32_t line_no);

    static size_t
    GetEnvironment (StringList &env);
};

} // namespace lldb_private

#endif  // #if defined(__cplusplus)
#endif  // liblldb_Host_h_
