//===-- Process.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Process_h_
#define liblldb_Process_h_

// C Includes
#include <limits.h>
#include <spawn.h>

// C++ Includes
#include <list>
#include <iosfwd>
#include <vector>

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/Broadcaster.h"
#include "lldb/Core/Communication.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Event.h"
#include "lldb/Core/RangeMap.h"
#include "lldb/Core/StringList.h"
#include "lldb/Core/ThreadSafeValue.h"
#include "lldb/Core/PluginInterface.h"
#include "lldb/Core/UserSettingsController.h"
#include "lldb/Breakpoint/BreakpointSiteList.h"
#include "lldb/Expression/ClangPersistentVariables.h"
#include "lldb/Expression/IRDynamicChecks.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/ReadWriteLock.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Target/ExecutionContextScope.h"
#include "lldb/Target/Memory.h"
#include "lldb/Target/ThreadList.h"
#include "lldb/Target/UnixSignals.h"
#include "lldb/Utility/PseudoTerminal.h"

namespace lldb_private {

//----------------------------------------------------------------------
// ProcessProperties
//----------------------------------------------------------------------
class ProcessProperties : public Properties
{
public:
    ProcessProperties(bool is_global);

    virtual
    ~ProcessProperties();
    
    bool
    GetDisableMemoryCache() const;

    Args
    GetExtraStartupCommands () const;

    void
    SetExtraStartupCommands (const Args &args);
    
    FileSpec
    GetPythonOSPluginPath () const;

    void
    SetPythonOSPluginPath (const FileSpec &file);
};

typedef STD_SHARED_PTR(ProcessProperties) ProcessPropertiesSP;

//----------------------------------------------------------------------
// ProcessInfo
//
// A base class for information for a process. This can be used to fill
// out information for a process prior to launching it, or it can be 
// used for an instance of a process and can be filled in with the 
// existing values for that process.
//----------------------------------------------------------------------
class ProcessInfo
{
public:
    ProcessInfo () :
        m_executable (),
        m_arguments (),
        m_environment (),
        m_uid (UINT32_MAX),
        m_gid (UINT32_MAX),
        m_arch(),
        m_pid (LLDB_INVALID_PROCESS_ID)
    {
    }
    
    ProcessInfo (const char *name,
                 const ArchSpec &arch,
                 lldb::pid_t pid) :
        m_executable (name, false),
        m_arguments (),
        m_environment(),
        m_uid (UINT32_MAX),
        m_gid (UINT32_MAX),
        m_arch (arch),
        m_pid (pid)
    {
    }
    
    void
    Clear ()
    {
        m_executable.Clear();
        m_arguments.Clear();
        m_environment.Clear();
        m_uid = UINT32_MAX;
        m_gid = UINT32_MAX;
        m_arch.Clear();
        m_pid = LLDB_INVALID_PROCESS_ID;
    }
    
    const char *
    GetName() const
    {
        return m_executable.GetFilename().GetCString();
    }
    
    size_t
    GetNameLength() const
    {
        return m_executable.GetFilename().GetLength();
    }
    
    FileSpec &
    GetExecutableFile ()
    {
        return m_executable;
    }

    void
    SetExecutableFile (const FileSpec &exe_file, bool add_exe_file_as_first_arg)
    {
        if (exe_file)
        {
            m_executable = exe_file;
            if (add_exe_file_as_first_arg)
            {
                char filename[PATH_MAX];
                if (exe_file.GetPath(filename, sizeof(filename)))
                    m_arguments.InsertArgumentAtIndex (0, filename);
            }
        }
        else
        {
            m_executable.Clear();
        }
    }

    const FileSpec &
    GetExecutableFile () const
    {
        return m_executable;
    }
    
    uint32_t
    GetUserID() const
    {
        return m_uid;
    }
    
    uint32_t
    GetGroupID() const
    {
        return m_gid;
    }
    
    bool
    UserIDIsValid () const
    {
        return m_uid != UINT32_MAX;
    }
    
    bool
    GroupIDIsValid () const
    {
        return m_gid != UINT32_MAX;
    }
    
    void
    SetUserID (uint32_t uid)
    {
        m_uid = uid;
    }
    
    void
    SetGroupID (uint32_t gid)
    {
        m_gid = gid;
    }
    
    ArchSpec &
    GetArchitecture ()
    {
        return m_arch;
    }
    
    const ArchSpec &
    GetArchitecture () const
    {
        return m_arch;
    }
    
    lldb::pid_t
    GetProcessID () const
    {
        return m_pid;
    }
    
    void
    SetProcessID (lldb::pid_t pid)
    {
        m_pid = pid;
    }
    
    bool
    ProcessIDIsValid() const
    {
        return m_pid != LLDB_INVALID_PROCESS_ID;
    }
    
    void
    Dump (Stream &s, Platform *platform) const;
    
    Args &
    GetArguments ()
    {
        return m_arguments;
    }
    
    const Args &
    GetArguments () const
    {
        return m_arguments;
    }
    
    const char *
    GetArg0 () const
    {
        if (m_arg0.empty())
            return NULL;
        return m_arg0.c_str();
    }
    
    void
    SetArg0 (const char *arg)
    {
        if (arg && arg[0])
            m_arg0.clear();
        else
            m_arg0 = arg;
    }
    
    void
    SetArguments (const Args& args, bool first_arg_is_executable);

    void
    SetArguments (char const **argv, bool first_arg_is_executable);
    
    Args &
    GetEnvironmentEntries ()
    {
        return m_environment;
    }
    
    const Args &
    GetEnvironmentEntries () const
    {
        return m_environment;
    }
    
protected:
    FileSpec m_executable;
    std::string m_arg0; // argv[0] if supported. If empty, then use m_executable.
                        // Not all process plug-ins support specifying an argv[0]
                        // that differs from the resolved platform executable
                        // (which is in m_executable)
    Args m_arguments;   // All program arguments except argv[0]
    Args m_environment;
    uint32_t m_uid;
    uint32_t m_gid;    
    ArchSpec m_arch;
    lldb::pid_t m_pid;
};

//----------------------------------------------------------------------
// ProcessInstanceInfo
//
// Describes an existing process and any discoverable information that
// pertains to that process.
//----------------------------------------------------------------------
class ProcessInstanceInfo : public ProcessInfo
{
public:
    ProcessInstanceInfo () :
        ProcessInfo (),
        m_euid (UINT32_MAX),
        m_egid (UINT32_MAX),
        m_parent_pid (LLDB_INVALID_PROCESS_ID)
    {
    }

    ProcessInstanceInfo (const char *name,
                 const ArchSpec &arch,
                 lldb::pid_t pid) :
        ProcessInfo (name, arch, pid),
        m_euid (UINT32_MAX),
        m_egid (UINT32_MAX),
        m_parent_pid (LLDB_INVALID_PROCESS_ID)
    {
    }
    
    void
    Clear ()
    {
        ProcessInfo::Clear();
        m_euid = UINT32_MAX;
        m_egid = UINT32_MAX;
        m_parent_pid = LLDB_INVALID_PROCESS_ID;
    }
    
    uint32_t
    GetEffectiveUserID() const
    {
        return m_euid;
    }

    uint32_t
    GetEffectiveGroupID() const
    {
        return m_egid;
    }
    
    bool
    EffectiveUserIDIsValid () const
    {
        return m_euid != UINT32_MAX;
    }

    bool
    EffectiveGroupIDIsValid () const
    {
        return m_egid != UINT32_MAX;
    }

    void
    SetEffectiveUserID (uint32_t uid)
    {
        m_euid = uid;
    }
    
    void
    SetEffectiveGroupID (uint32_t gid)
    {
        m_egid = gid;
    }

    lldb::pid_t
    GetParentProcessID () const
    {
        return m_parent_pid;
    }
    
    void
    SetParentProcessID (lldb::pid_t pid)
    {
        m_parent_pid = pid;
    }
    
    bool
    ParentProcessIDIsValid() const
    {
        return m_parent_pid != LLDB_INVALID_PROCESS_ID;
    }
    
    void
    Dump (Stream &s, Platform *platform) const;

    static void
    DumpTableHeader (Stream &s, Platform *platform, bool show_args, bool verbose);

    void
    DumpAsTableRow (Stream &s, Platform *platform, bool show_args, bool verbose) const;
    
protected:
    uint32_t m_euid;
    uint32_t m_egid;    
    lldb::pid_t m_parent_pid;
};

    
//----------------------------------------------------------------------
// ProcessLaunchInfo
//
// Describes any information that is required to launch a process.
//----------------------------------------------------------------------

class ProcessLaunchInfo : public ProcessInfo
{
public:

    class FileAction
    {
    public:
        enum Action
        {
            eFileActionNone,
            eFileActionClose,
            eFileActionDuplicate,
            eFileActionOpen
        };
        

        FileAction () :
            m_action (eFileActionNone),
            m_fd (-1),
            m_arg (-1),
            m_path ()
        {
        }

        void
        Clear()
        {
            m_action = eFileActionNone;
            m_fd = -1;
            m_arg = -1;
            m_path.clear();
        }

        bool
        Close (int fd);

        bool
        Duplicate (int fd, int dup_fd);

        bool
        Open (int fd, const char *path, bool read, bool write);
        
        static bool
        AddPosixSpawnFileAction (posix_spawn_file_actions_t *file_actions,
                                 const FileAction *info,
                                 Log *log, 
                                 Error& error);

        int
        GetFD () const
        {
            return m_fd;
        }

        Action 
        GetAction () const
        {
            return m_action;
        }
        
        int 
        GetActionArgument () const
        {
            return m_arg;
        }
        
        const char *
        GetPath () const
        {
            if (m_path.empty())
                return NULL;
            return m_path.c_str();
        }

    protected:
        Action m_action;    // The action for this file
        int m_fd;           // An existing file descriptor
        int m_arg;          // oflag for eFileActionOpen*, dup_fd for eFileActionDuplicate
        std::string m_path; // A file path to use for opening after fork or posix_spawn
    };
    
    ProcessLaunchInfo () :
        ProcessInfo(),
        m_working_dir (),
        m_plugin_name (),
        m_shell (),
        m_flags (0),
        m_file_actions (), 
        m_pty (),
        m_resume_count (0),
        m_monitor_callback (NULL),
        m_monitor_callback_baton (NULL),
        m_monitor_signals (false)
    {
    }

    ProcessLaunchInfo (const char *stdin_path,
                       const char *stdout_path,
                       const char *stderr_path,
                       const char *working_directory,
                       uint32_t launch_flags) :
        ProcessInfo(),
        m_working_dir (),
        m_plugin_name (),
        m_shell (),
        m_flags (launch_flags),
        m_file_actions (), 
        m_pty (),
        m_resume_count (0),
        m_monitor_callback (NULL),
        m_monitor_callback_baton (NULL),
        m_monitor_signals (false)
    {
        if (stdin_path)
        {
            ProcessLaunchInfo::FileAction file_action;
            const bool read = true;
            const bool write = false;
            if (file_action.Open(STDIN_FILENO, stdin_path, read, write))
                AppendFileAction (file_action);
        }
        if (stdout_path)
        {
            ProcessLaunchInfo::FileAction file_action;
            const bool read = false;
            const bool write = true;
            if (file_action.Open(STDOUT_FILENO, stdout_path, read, write))
                AppendFileAction (file_action);
        }
        if (stderr_path)
        {
            ProcessLaunchInfo::FileAction file_action;
            const bool read = false;
            const bool write = true;
            if (file_action.Open(STDERR_FILENO, stderr_path, read, write))
                AppendFileAction (file_action);
        }
        if (working_directory)
            SetWorkingDirectory(working_directory);        
    }

    void
    AppendFileAction (const FileAction &info)
    {
        m_file_actions.push_back(info);
    }

    bool
    AppendCloseFileAction (int fd)
    {
        FileAction file_action;
        if (file_action.Close (fd))
        {
            AppendFileAction (file_action);
            return true;
        }
        return false;
    }

    bool
    AppendDuplicateFileAction (int fd, int dup_fd)
    {
        FileAction file_action;
        if (file_action.Duplicate (fd, dup_fd))
        {
            AppendFileAction (file_action);
            return true;
        }
        return false;
    }

    bool
    AppendOpenFileAction (int fd, const char *path, bool read, bool write)
    {
        FileAction file_action;
        if (file_action.Open (fd, path, read, write))
        {
            AppendFileAction (file_action);
            return true;
        }
        return false;
    }

    bool
    AppendSuppressFileAction (int fd, bool read, bool write)
    {
        FileAction file_action;
        if (file_action.Open (fd, "/dev/null", read, write))
        {
            AppendFileAction (file_action);
            return true;
        }
        return false;
    }
    
    void
    FinalizeFileActions (Target *target, 
                         bool default_to_use_pty);

    size_t
    GetNumFileActions () const
    {
        return m_file_actions.size();
    }
    
    const FileAction *
    GetFileActionAtIndex (size_t idx) const
    {
        if (idx < m_file_actions.size())
            return &m_file_actions[idx];
        return NULL;
    }

    const FileAction *
    GetFileActionForFD (int fd) const
    {
        for (uint32_t idx=0, count=m_file_actions.size(); idx < count; ++idx)
        {
            if (m_file_actions[idx].GetFD () == fd)
                return &m_file_actions[idx];
        }
        return NULL;
    }

    Flags &
    GetFlags ()
    {
        return m_flags;
    }

    const Flags &
    GetFlags () const
    {
        return m_flags;
    }
    
    const char *
    GetWorkingDirectory () const
    {
        if (m_working_dir.empty())
            return NULL;
        return m_working_dir.c_str();
    }

    void
    SetWorkingDirectory (const char *working_dir)
    {
        if (working_dir && working_dir[0])
            m_working_dir.assign (working_dir);
        else
            m_working_dir.clear();
    }

    void
    SwapWorkingDirectory (std::string &working_dir)
    {
        m_working_dir.swap (working_dir);
    }


    const char *
    GetProcessPluginName () const
    {
        if (m_plugin_name.empty())
            return NULL;
        return m_plugin_name.c_str();
    }

    void
    SetProcessPluginName (const char *plugin)
    {
        if (plugin && plugin[0])
            m_plugin_name.assign (plugin);
        else
            m_plugin_name.clear();
    }
    
    const char *
    GetShell () const
    {
        if (m_shell.empty())
            return NULL;
        return m_shell.c_str();
    }

    void
    SetShell (const char * path)
    {
        if (path && path[0])
        {
            m_shell.assign (path);
            m_flags.Set (lldb::eLaunchFlagLaunchInShell);
        }
        else
        {
            m_shell.clear();
            m_flags.Clear (lldb::eLaunchFlagLaunchInShell);
        }
    }

    uint32_t
    GetResumeCount () const
    {
        return m_resume_count;
    }
    
    void
    SetResumeCount (uint32_t c)
    {
        m_resume_count = c;
    }
    
    bool
    GetLaunchInSeparateProcessGroup ()
    {
        return m_flags.Test(lldb::eLaunchFlagLaunchInSeparateProcessGroup);
    }
    
    void
    SetLaunchInSeparateProcessGroup (bool separate)
    {
        if (separate)
            m_flags.Set(lldb::eLaunchFlagLaunchInSeparateProcessGroup);
        else
            m_flags.Clear (lldb::eLaunchFlagLaunchInSeparateProcessGroup);

    }

    void
    Clear ()
    {
        ProcessInfo::Clear();
        m_working_dir.clear();
        m_plugin_name.clear();
        m_shell.clear();
        m_flags.Clear();
        m_file_actions.clear();
        m_resume_count = 0;
    }

    bool
    ConvertArgumentsForLaunchingInShell (Error &error,
                                         bool localhost,
                                         bool will_debug,
                                         bool first_arg_is_full_shell_command);
    
    void
    SetMonitorProcessCallback (Host::MonitorChildProcessCallback callback, 
                               void *baton, 
                               bool monitor_signals)
    {
        m_monitor_callback = callback;
        m_monitor_callback_baton = baton;
        m_monitor_signals = monitor_signals;
    }

    bool
    MonitorProcess () const
    {
        if (m_monitor_callback && ProcessIDIsValid())
        {
            Host::StartMonitoringChildProcess (m_monitor_callback,
                                               m_monitor_callback_baton,
                                               GetProcessID(), 
                                               m_monitor_signals);
            return true;
        }
        return false;
    }
    
    lldb_utility::PseudoTerminal &
    GetPTY ()
    {
        return m_pty;
    }

protected:
    std::string m_working_dir;
    std::string m_plugin_name;
    std::string m_shell;
    Flags m_flags;       // Bitwise OR of bits from lldb::LaunchFlags
    std::vector<FileAction> m_file_actions; // File actions for any other files
    lldb_utility::PseudoTerminal m_pty;
    uint32_t m_resume_count; // How many times do we resume after launching
    Host::MonitorChildProcessCallback m_monitor_callback;
    void *m_monitor_callback_baton;
    bool m_monitor_signals;

};

//----------------------------------------------------------------------
// ProcessLaunchInfo
//
// Describes any information that is required to launch a process.
//----------------------------------------------------------------------
    
class ProcessAttachInfo : public ProcessInstanceInfo
{
public:
    ProcessAttachInfo() :
        ProcessInstanceInfo(),
        m_plugin_name (),
        m_resume_count (0),
        m_wait_for_launch (false),
        m_ignore_existing (true),
        m_continue_once_attached (false)
    {
    }

    ProcessAttachInfo (const ProcessLaunchInfo &launch_info) :
        ProcessInstanceInfo(),
        m_plugin_name (),
        m_resume_count (0),
        m_wait_for_launch (false),
        m_ignore_existing (true),
        m_continue_once_attached (false)
    {
        ProcessInfo::operator= (launch_info);
        SetProcessPluginName (launch_info.GetProcessPluginName());
        SetResumeCount (launch_info.GetResumeCount());
    }
    
    bool
    GetWaitForLaunch () const
    {
        return m_wait_for_launch;
    }
    
    void
    SetWaitForLaunch (bool b)
    {
        m_wait_for_launch = b;
    }

    bool
    GetIgnoreExisting () const
    {
        return m_ignore_existing;
    }
    
    void
    SetIgnoreExisting (bool b)
    {
        m_ignore_existing = b;
    }

    bool
    GetContinueOnceAttached () const
    {
        return m_continue_once_attached;
    }
    
    void
    SetContinueOnceAttached (bool b)
    {
        m_continue_once_attached = b;
    }

    uint32_t
    GetResumeCount () const
    {
        return m_resume_count;
    }
    
    void
    SetResumeCount (uint32_t c)
    {
        m_resume_count = c;
    }
    
    const char *
    GetProcessPluginName () const
    {
        if (m_plugin_name.empty())
            return NULL;
        return m_plugin_name.c_str();
    }
    
    void
    SetProcessPluginName (const char *plugin)
    {
        if (plugin && plugin[0])
            m_plugin_name.assign (plugin);
        else
            m_plugin_name.clear();
    }

    void
    Clear ()
    {
        ProcessInstanceInfo::Clear();
        m_plugin_name.clear();
        m_resume_count = 0;
        m_wait_for_launch = false;
        m_ignore_existing = true;
        m_continue_once_attached = false;
    }

    bool
    ProcessInfoSpecified () const
    {
        if (GetExecutableFile())
            return true;
        if (GetProcessID() != LLDB_INVALID_PROCESS_ID)
            return true;
        if (GetParentProcessID() != LLDB_INVALID_PROCESS_ID)
            return true;
        return false;
    }
protected:
    std::string m_plugin_name;
    uint32_t m_resume_count; // How many times do we resume after launching
    bool m_wait_for_launch;
    bool m_ignore_existing;
    bool m_continue_once_attached; // Supports the use-case scenario of immediately continuing the process once attached.
};

class ProcessLaunchCommandOptions : public Options
{
public:
    
    ProcessLaunchCommandOptions (CommandInterpreter &interpreter) :
        Options(interpreter)
    {
        // Keep default values of all options in one place: OptionParsingStarting ()
        OptionParsingStarting ();
    }
    
    ~ProcessLaunchCommandOptions ()
    {
    }
    
    Error
    SetOptionValue (uint32_t option_idx, const char *option_arg);
    
    void
    OptionParsingStarting ()
    {
        launch_info.Clear();
    }
    
    const OptionDefinition*
    GetDefinitions ()
    {
        return g_option_table;
    }
    
    // Options table: Required for subclasses of Options.
    
    static OptionDefinition g_option_table[];
    
    // Instance variables to hold the values for command options.
    
    ProcessLaunchInfo launch_info;
};

//----------------------------------------------------------------------
// ProcessInstanceInfoMatch
//
// A class to help matching one ProcessInstanceInfo to another.
//----------------------------------------------------------------------

class ProcessInstanceInfoMatch
{
public:
    ProcessInstanceInfoMatch () :
        m_match_info (),
        m_name_match_type (eNameMatchIgnore),
        m_match_all_users (false)
    {
    }

    ProcessInstanceInfoMatch (const char *process_name, 
                              NameMatchType process_name_match_type) :
        m_match_info (),
        m_name_match_type (process_name_match_type),
        m_match_all_users (false)
    {
        m_match_info.GetExecutableFile().SetFile(process_name, false);
    }

    ProcessInstanceInfo &
    GetProcessInfo ()
    {
        return m_match_info;
    }

    const ProcessInstanceInfo &
    GetProcessInfo () const
    {
        return m_match_info;
    }
    
    bool
    GetMatchAllUsers () const
    {
        return m_match_all_users;
    }

    void
    SetMatchAllUsers (bool b)
    {
        m_match_all_users = b;
    }

    NameMatchType
    GetNameMatchType () const
    {
        return m_name_match_type;
    }

    void
    SetNameMatchType (NameMatchType name_match_type)
    {
        m_name_match_type = name_match_type;
    }
    
    bool
    NameMatches (const char *process_name) const;

    bool
    Matches (const ProcessInstanceInfo &proc_info) const;

    bool
    MatchAllProcesses () const;
    void
    Clear ();

protected:
    ProcessInstanceInfo m_match_info;
    NameMatchType m_name_match_type;
    bool m_match_all_users;
};

class ProcessInstanceInfoList
{
public:
    ProcessInstanceInfoList () :
        m_infos()
    {
    }
    
    void
    Clear()
    {
        m_infos.clear();
    }
    
    uint32_t
    GetSize()
    {
        return m_infos.size();
    }
    
    void
    Append (const ProcessInstanceInfo &info)
    {
        m_infos.push_back (info);
    }

    const char *
    GetProcessNameAtIndex (uint32_t idx)
    {
        if (idx < m_infos.size())
            return m_infos[idx].GetName();
        return NULL;
    }

    size_t
    GetProcessNameLengthAtIndex (uint32_t idx)
    {
        if (idx < m_infos.size())
            return m_infos[idx].GetNameLength();
        return 0;
    }

    lldb::pid_t
    GetProcessIDAtIndex (uint32_t idx)
    {
        if (idx < m_infos.size())
            return m_infos[idx].GetProcessID();
        return 0;
    }

    bool
    GetInfoAtIndex (uint32_t idx, ProcessInstanceInfo &info)
    {
        if (idx < m_infos.size())
        {
            info = m_infos[idx];
            return true;
        }
        return false;
    }
    
    // You must ensure "idx" is valid before calling this function
    const ProcessInstanceInfo &
    GetProcessInfoAtIndex (uint32_t idx) const
    {
        assert (idx < m_infos.size());
        return m_infos[idx];
    }
    
protected:
    typedef std::vector<ProcessInstanceInfo> collection;
    collection m_infos;
};


// This class tracks the Modification state of the process.  Things that can currently modify
// the program are running the program (which will up the StopID) and writing memory (which
// will up the MemoryID.)  
// FIXME: Should we also include modification of register states?

class ProcessModID
{
friend bool operator== (const ProcessModID &lhs, const ProcessModID &rhs);   
public:
    ProcessModID () : 
        m_stop_id (0),
        m_resume_id (0), 
        m_memory_id (0),
        m_last_user_expression_resume (0),
        m_running_user_expression (false)
    {}
    
    ProcessModID (const ProcessModID &rhs) :
        m_stop_id (rhs.m_stop_id),
        m_memory_id (rhs.m_memory_id)
    {}
    
    const ProcessModID & operator= (const ProcessModID &rhs)
    {
        if (this != &rhs)
        {
            m_stop_id = rhs.m_stop_id;
            m_memory_id = rhs.m_memory_id;
        }
        return *this;
    }
    
    ~ProcessModID () {}
    
    void BumpStopID () { 
        m_stop_id++; 
    }
    
    void BumpMemoryID () { m_memory_id++; }
    
    void BumpResumeID () {
        m_resume_id++;
        if (m_running_user_expression > 0)
            m_last_user_expression_resume = m_resume_id;
    }
    
    uint32_t GetStopID() const { return m_stop_id; }
    uint32_t GetMemoryID () const { return m_memory_id; }
    uint32_t GetResumeID () const { return m_resume_id; }
    uint32_t GetLastUserExpressionResumeID () const { return m_last_user_expression_resume; }
    
    bool MemoryIDEqual (const ProcessModID &compare) const
    {
        return m_memory_id == compare.m_memory_id;
    }
    
    bool StopIDEqual (const ProcessModID &compare) const
    {
        return m_stop_id == compare.m_stop_id;
    }
    
    void SetInvalid ()
    {
        m_stop_id = UINT32_MAX;
    }
    
    bool IsValid () const
    {
        return m_stop_id != UINT32_MAX;
    }
    
    bool
    IsLastResumeForUserExpression () const
    {
        return m_resume_id == m_last_user_expression_resume;
    }
    
    void
    SetRunningUserExpression (bool on)
    {
        // REMOVEME printf ("Setting running user expression %s at resume id %d - value: %d.\n", on ? "on" : "off", m_resume_id, m_running_user_expression);
        if (on)
            m_running_user_expression++;
        else
            m_running_user_expression--;
    }
    
private:
    uint32_t m_stop_id;
    uint32_t m_resume_id;
    uint32_t m_memory_id;
    uint32_t m_last_user_expression_resume;
    uint32_t m_running_user_expression;
};
inline bool operator== (const ProcessModID &lhs, const ProcessModID &rhs)
{
    if (lhs.StopIDEqual (rhs)
        && lhs.MemoryIDEqual (rhs))
        return true;
    else
        return false;
}

inline bool operator!= (const ProcessModID &lhs, const ProcessModID &rhs)
{
    if (!lhs.StopIDEqual (rhs)
        || !lhs.MemoryIDEqual (rhs))
        return true;
    else
        return false;
}
    
class MemoryRegionInfo
{
public:
    typedef Range<lldb::addr_t, lldb::addr_t> RangeType;

    enum OptionalBool {
        eDontKnow  = -1,
        eNo         = 0,
        eYes        = 1
    };

    MemoryRegionInfo () :
        m_range (),
        m_read (eDontKnow),
        m_write (eDontKnow),
        m_execute (eDontKnow)
    {
    }

    ~MemoryRegionInfo ()
    {
    }

    RangeType &
    GetRange()
    {
        return m_range;
    }

    void
    Clear()
    {
        m_range.Clear();
        m_read = m_write = m_execute = eDontKnow;
    }

    const RangeType &
    GetRange() const
    {
        return m_range;
    }

    OptionalBool
    GetReadable () const
    {
        return m_read;
    }

    OptionalBool
    GetWritable () const
    {
        return m_write;
    }

    OptionalBool
    GetExecutable () const
    {
        return m_execute;
    }

    void
    SetReadable (OptionalBool val)
    {
        m_read = val;
    }

    void
    SetWritable (OptionalBool val)
    {
        m_write = val;
    }

    void
    SetExecutable (OptionalBool val)
    {
        m_execute = val;
    }

protected:
    RangeType m_range;
    OptionalBool m_read;
    OptionalBool m_write;
    OptionalBool m_execute;
};

//----------------------------------------------------------------------
/// @class Process Process.h "lldb/Target/Process.h"
/// @brief A plug-in interface definition class for debugging a process.
//----------------------------------------------------------------------
class Process :
    public STD_ENABLE_SHARED_FROM_THIS(Process),
    public ProcessProperties,
    public UserID,
    public Broadcaster,
    public ExecutionContextScope,
    public PluginInterface
{
friend class ThreadList;
friend class ClangFunction; // For WaitForStateChangeEventsPrivate
friend class CommandObjectProcessLaunch;
friend class ProcessEventData;
friend class CommandObjectBreakpointCommand;
friend class StopInfo;

public:

    //------------------------------------------------------------------
    /// Broadcaster event bits definitions.
    //------------------------------------------------------------------
    enum
    {
        eBroadcastBitStateChanged   = (1 << 0),
        eBroadcastBitInterrupt      = (1 << 1),
        eBroadcastBitSTDOUT         = (1 << 2),
        eBroadcastBitSTDERR         = (1 << 3)
    };

    enum
    {
        eBroadcastInternalStateControlStop = (1<<0),
        eBroadcastInternalStateControlPause = (1<<1),
        eBroadcastInternalStateControlResume = (1<<2)
    };
    
    typedef Range<lldb::addr_t, lldb::addr_t> LoadRange;
    // We use a read/write lock to allow on or more clients to
    // access the process state while the process is stopped (reader).
    // We lock the write lock to control access to the process
    // while it is running (readers, or clients that want the process
    // stopped can block waiting for the process to stop, or just
    // try to lock it to see if they can immediately access the stopped
    // process. If the try read lock fails, then the process is running.
    typedef ReadWriteLock::ReadLocker StopLocker;
    typedef ReadWriteLock::WriteLocker RunLocker;

    // These two functions fill out the Broadcaster interface:
    
    static ConstString &GetStaticBroadcasterClass ();

    virtual ConstString &GetBroadcasterClass() const
    {
        return GetStaticBroadcasterClass();
    }

    //------------------------------------------------------------------
    /// A notification structure that can be used by clients to listen
    /// for changes in a process's lifetime.
    ///
    /// @see RegisterNotificationCallbacks (const Notifications&)
    /// @see UnregisterNotificationCallbacks (const Notifications&)
    //------------------------------------------------------------------
#ifndef SWIG
    typedef struct
    {
        void *baton;
        void (*initialize)(void *baton, Process *process);
        void (*process_state_changed) (void *baton, Process *process, lldb::StateType state);
    } Notifications;

    class ProcessEventData :
        public EventData
    {
        friend class Process;
        
        public:
            ProcessEventData ();
            ProcessEventData (const lldb::ProcessSP &process, lldb::StateType state);

            virtual ~ProcessEventData();

            static const ConstString &
            GetFlavorString ();

            virtual const ConstString &
            GetFlavor () const;

            const lldb::ProcessSP &
            GetProcessSP() const
            {
                return m_process_sp;
            }
            lldb::StateType
            GetState() const
            {
                return m_state;
            }
            bool
            GetRestarted () const
            {
                return m_restarted;
            }
            bool
            GetInterrupted () const
            {
                return m_interrupted;
            }

            virtual void
            Dump (Stream *s) const;

            virtual void
            DoOnRemoval (Event *event_ptr);

            static const Process::ProcessEventData *
            GetEventDataFromEvent (const Event *event_ptr);

            static lldb::ProcessSP
            GetProcessFromEvent (const Event *event_ptr);

            static lldb::StateType
            GetStateFromEvent (const Event *event_ptr);

            static bool
            GetRestartedFromEvent (const Event *event_ptr);

            static void
            SetRestartedInEvent (Event *event_ptr, bool new_value);

            static bool
            GetInterruptedFromEvent (const Event *event_ptr);

            static void
            SetInterruptedInEvent (Event *event_ptr, bool new_value);

            static bool
            SetUpdateStateOnRemoval (Event *event_ptr);

       private:

            void
            SetUpdateStateOnRemoval()
            {
                m_update_state++;
            }
            void
            SetRestarted (bool new_value)
            {
                m_restarted = new_value;
            }
            void
            SetInterrupted (bool new_value)
            {
                m_interrupted = new_value;
            }

            lldb::ProcessSP m_process_sp;
            lldb::StateType m_state;
            bool m_restarted;  // For "eStateStopped" events, this is true if the target was automatically restarted.
            int m_update_state;
            bool m_interrupted;
            DISALLOW_COPY_AND_ASSIGN (ProcessEventData);

    };

#endif

    static void
    SettingsInitialize ();

    static void
    SettingsTerminate ();
    
    static const ProcessPropertiesSP &
    GetGlobalProperties();

    //------------------------------------------------------------------
    /// Construct with a shared pointer to a target, and the Process listener.
    //------------------------------------------------------------------
    Process(Target &target, Listener &listener);

    //------------------------------------------------------------------
    /// Destructor.
    ///
    /// The destructor is virtual since this class is designed to be
    /// inherited from by the plug-in instance.
    //------------------------------------------------------------------
    virtual
    ~Process();

    //------------------------------------------------------------------
    /// Find a Process plug-in that can debug \a module using the
    /// currently selected architecture.
    ///
    /// Scans all loaded plug-in interfaces that implement versions of
    /// the Process plug-in interface and returns the first instance
    /// that can debug the file.
    ///
    /// @param[in] module_sp
    ///     The module shared pointer that this process will debug.
    ///
    /// @param[in] plugin_name
    ///     If NULL, select the best plug-in for the binary. If non-NULL
    ///     then look for a plugin whose PluginInfo's name matches
    ///     this string.
    ///
    /// @see Process::CanDebug ()
    //------------------------------------------------------------------
    static lldb::ProcessSP
    FindPlugin (Target &target, 
                const char *plugin_name, 
                Listener &listener, 
                const FileSpec *crash_file_path);



    //------------------------------------------------------------------
    /// Static function that can be used with the \b host function
    /// Host::StartMonitoringChildProcess ().
    ///
    /// This function can be used by lldb_private::Process subclasses
    /// when they want to watch for a local process and have its exit
    /// status automatically set when the host child process exits.
    /// Subclasses should call Host::StartMonitoringChildProcess ()
    /// with:
    ///     callback = Process::SetHostProcessExitStatus
    ///     callback_baton = NULL
    ///     pid = Process::GetID()
    ///     monitor_signals = false
    //------------------------------------------------------------------
    static bool
    SetProcessExitStatus (void *callback_baton,   // The callback baton which should be set to NULL
                          lldb::pid_t pid,        // The process ID we want to monitor
                          bool exited,
                          int signo,              // Zero for no signal
                          int status);            // Exit value of process if signal is zero

    lldb::ByteOrder
    GetByteOrder () const;
    
    uint32_t
    GetAddressByteSize () const;

    //------------------------------------------------------------------
    /// Check if a plug-in instance can debug the file in \a module.
    ///
    /// Each plug-in is given a chance to say whether it can debug
    /// the file in \a module. If the Process plug-in instance can
    /// debug a file on the current system, it should return \b true.
    ///
    /// @return
    ///     Returns \b true if this Process plug-in instance can
    ///     debug the executable, \b false otherwise.
    //------------------------------------------------------------------
    virtual bool
    CanDebug (Target &target,
              bool plugin_specified_by_name) = 0;


    //------------------------------------------------------------------
    /// This object is about to be destroyed, do any necessary cleanup.
    ///
    /// Subclasses that override this method should always call this
    /// superclass method.
    //------------------------------------------------------------------
    virtual void
    Finalize();
    
    
    //------------------------------------------------------------------
    /// Return whether this object is valid (i.e. has not been finalized.)
    ///
    /// @return
    ///     Returns \b true if this Process has not been finalized
    ///     and \b false otherwise.
    //------------------------------------------------------------------
    bool
    IsValid() const
    {
        return !m_finalize_called;
    }

    //------------------------------------------------------------------
    /// Return a multi-word command object that can be used to expose
    /// plug-in specific commands.
    ///
    /// This object will be used to resolve plug-in commands and can be
    /// triggered by a call to:
    ///
    ///     (lldb) process commmand <args>
    ///
    /// @return
    ///     A CommandObject which can be one of the concrete subclasses
    ///     of CommandObject like CommandObjectRaw, CommandObjectParsed,
    ///     or CommandObjectMultiword.
    //------------------------------------------------------------------
    virtual CommandObject *
    GetPluginCommandObject()
    {
        return NULL;
    }

    //------------------------------------------------------------------
    /// Launch a new process.
    ///
    /// Launch a new process by spawning a new process using the
    /// target object's executable module's file as the file to launch.
    /// Arguments are given in \a argv, and the environment variables
    /// are in \a envp. Standard input and output files can be
    /// optionally re-directed to \a stdin_path, \a stdout_path, and
    /// \a stderr_path.
    ///
    /// This function is not meant to be overridden by Process
    /// subclasses. It will first call Process::WillLaunch (Module *)
    /// and if that returns \b true, Process::DoLaunch (Module*,
    /// char const *[],char const *[],const char *,const char *,
    /// const char *) will be called to actually do the launching. If
    /// DoLaunch returns \b true, then Process::DidLaunch() will be
    /// called.
    ///
    /// @param[in] argv
    ///     The argument array.
    ///
    /// @param[in] envp
    ///     The environment array.
    ///
    /// @param[in] launch_flags
    ///     Flags to modify the launch (@see lldb::LaunchFlags)
    ///
    /// @param[in] stdin_path
    ///     The path to use when re-directing the STDIN of the new
    ///     process. If all stdXX_path arguments are NULL, a pseudo
    ///     terminal will be used.
    ///
    /// @param[in] stdout_path
    ///     The path to use when re-directing the STDOUT of the new
    ///     process. If all stdXX_path arguments are NULL, a pseudo
    ///     terminal will be used.
    ///
    /// @param[in] stderr_path
    ///     The path to use when re-directing the STDERR of the new
    ///     process. If all stdXX_path arguments are NULL, a pseudo
    ///     terminal will be used.
    ///
    /// @param[in] working_directory
    ///     The working directory to have the child process run in
    ///
    /// @return
    ///     An error object. Call GetID() to get the process ID if
    ///     the error object is success.
    //------------------------------------------------------------------
    virtual Error
    Launch (const ProcessLaunchInfo &launch_info);

    virtual Error
    LoadCore ();

    virtual Error
    DoLoadCore ()
    {
        Error error;
        error.SetErrorStringWithFormat("error: %s does not support loading core files.", GetShortPluginName());
        return error;
    }
    
    //------------------------------------------------------------------
    /// Get the dynamic loader plug-in for this process. 
    ///
    /// The default action is to let the DynamicLoader plug-ins check
    /// the main executable and the DynamicLoader will select itself
    /// automatically. Subclasses can override this if inspecting the
    /// executable is not desired, or if Process subclasses can only
    /// use a specific DynamicLoader plug-in.
    //------------------------------------------------------------------
    virtual DynamicLoader *
    GetDynamicLoader ();

    //------------------------------------------------------------------
    /// Attach to an existing process using the process attach info.
    ///
    /// This function is not meant to be overridden by Process
    /// subclasses. It will first call WillAttach (lldb::pid_t)
    /// or WillAttach (const char *), and if that returns \b 
    /// true, DoAttach (lldb::pid_t) or DoAttach (const char *) will
    /// be called to actually do the attach. If DoAttach returns \b
    /// true, then Process::DidAttach() will be called.
    ///
    /// @param[in] pid
    ///     The process ID that we should attempt to attach to.
    ///
    /// @return
    ///     Returns \a pid if attaching was successful, or
    ///     LLDB_INVALID_PROCESS_ID if attaching fails.
    //------------------------------------------------------------------
    virtual Error
    Attach (ProcessAttachInfo &attach_info);

    //------------------------------------------------------------------
    /// Attach to a remote system via a URL
    ///
    /// @param[in] strm
    ///     A stream where output intended for the user
    ///     (if the driver has a way to display that) generated during
    ///     the connection.  This may be NULL if no output is needed.A
    ///
    /// @param[in] remote_url
    ///     The URL format that we are connecting to.
    ///
    /// @return
    ///     Returns an error object.
    //------------------------------------------------------------------
    virtual Error
    ConnectRemote (Stream *strm, const char *remote_url);

    bool
    GetShouldDetach () const
    {
        return m_should_detach;
    }

    void
    SetShouldDetach (bool b)
    {
        m_should_detach = b;
    }

    //------------------------------------------------------------------
    /// Get the image information address for the current process.
    ///
    /// Some runtimes have system functions that can help dynamic
    /// loaders locate the dynamic loader information needed to observe
    /// shared libraries being loaded or unloaded. This function is
    /// in the Process interface (as opposed to the DynamicLoader
    /// interface) to ensure that remote debugging can take advantage of
    /// this functionality.
    ///
    /// @return
    ///     The address of the dynamic loader information, or
    ///     LLDB_INVALID_ADDRESS if this is not supported by this
    ///     interface.
    //------------------------------------------------------------------
    virtual lldb::addr_t
    GetImageInfoAddress ();

    //------------------------------------------------------------------
    /// Load a shared library into this process.
    ///
    /// Try and load a shared library into the current process. This
    /// call might fail in the dynamic loader plug-in says it isn't safe
    /// to try and load shared libraries at the moment.
    ///
    /// @param[in] image_spec
    ///     The image file spec that points to the shared library that
    ///     you want to load.
    ///
    /// @param[out] error
    ///     An error object that gets filled in with any errors that
    ///     might occur when trying to load the shared library.
    ///
    /// @return
    ///     A token that represents the shared library that can be
    ///     later used to unload the shared library. A value of
    ///     LLDB_INVALID_IMAGE_TOKEN will be returned if the shared
    ///     library can't be opened.
    //------------------------------------------------------------------
    virtual uint32_t
    LoadImage (const FileSpec &image_spec, Error &error);

    virtual Error
    UnloadImage (uint32_t image_token);

    //------------------------------------------------------------------
    /// Register for process and thread notifications.
    ///
    /// Clients can register nofication callbacks by filling out a
    /// Process::Notifications structure and calling this function.
    ///
    /// @param[in] callbacks
    ///     A structure that contains the notification baton and
    ///     callback functions.
    ///
    /// @see Process::Notifications
    //------------------------------------------------------------------
#ifndef SWIG
    void
    RegisterNotificationCallbacks (const Process::Notifications& callbacks);
#endif
    //------------------------------------------------------------------
    /// Unregister for process and thread notifications.
    ///
    /// Clients can unregister nofication callbacks by passing a copy of
    /// the original baton and callbacks in \a callbacks.
    ///
    /// @param[in] callbacks
    ///     A structure that contains the notification baton and
    ///     callback functions.
    ///
    /// @return
    ///     Returns \b true if the notification callbacks were
    ///     successfully removed from the process, \b false otherwise.
    ///
    /// @see Process::Notifications
    //------------------------------------------------------------------
#ifndef SWIG
    bool
    UnregisterNotificationCallbacks (const Process::Notifications& callbacks);
#endif
    //==================================================================
    // Built in Process Control functions
    //==================================================================
    //------------------------------------------------------------------
    /// Resumes all of a process's threads as configured using the
    /// Thread run control functions.
    ///
    /// Threads for a process should be updated with one of the run
    /// control actions (resume, step, or suspend) that they should take
    /// when the process is resumed. If no run control action is given
    /// to a thread it will be resumed by default.
    ///
    /// This function is not meant to be overridden by Process
    /// subclasses. This function will take care of disabling any
    /// breakpoints that threads may be stopped at, single stepping, and
    /// re-enabling breakpoints, and enabling the basic flow control
    /// that the plug-in instances need not worry about.
    ///
    /// N.B. This function also sets the Write side of the Run Lock,
    /// which is unset when the corresponding stop event is pulled off
    /// the Public Event Queue.  If you need to resume the process without
    /// setting the Run Lock, use PrivateResume (though you should only do
    /// that from inside the Process class.
    ///
    /// @return
    ///     Returns an error object.
    ///
    /// @see Thread:Resume()
    /// @see Thread:Step()
    /// @see Thread:Suspend()
    //------------------------------------------------------------------
    Error
    Resume();
    
    //------------------------------------------------------------------
    /// Halts a running process.
    ///
    /// This function is not meant to be overridden by Process
    /// subclasses.
    /// If the process is successfully halted, a eStateStopped
    /// process event with GetInterrupted will be broadcast.  If false, we will
    /// halt the process with no events generated by the halt.
    ///
    /// @return
    ///     Returns an error object.  If the error is empty, the process is halted.
    ///     otherwise the halt has failed.
    //------------------------------------------------------------------
    Error
    Halt ();

    //------------------------------------------------------------------
    /// Detaches from a running or stopped process.
    ///
    /// This function is not meant to be overridden by Process
    /// subclasses.
    ///
    /// @return
    ///     Returns an error object.
    //------------------------------------------------------------------
    Error
    Detach ();

    //------------------------------------------------------------------
    /// Kills the process and shuts down all threads that were spawned
    /// to track and monitor the process.
    ///
    /// This function is not meant to be overridden by Process
    /// subclasses.
    ///
    /// @return
    ///     Returns an error object.
    //------------------------------------------------------------------
    Error
    Destroy();

    //------------------------------------------------------------------
    /// Sends a process a UNIX signal \a signal.
    ///
    /// This function is not meant to be overridden by Process
    /// subclasses.
    ///
    /// @return
    ///     Returns an error object.
    //------------------------------------------------------------------
    Error
    Signal (int signal);

    virtual UnixSignals &
    GetUnixSignals ()
    {
        return m_unix_signals;
    }

    //==================================================================
    // Plug-in Process Control Overrides
    //==================================================================

    //------------------------------------------------------------------
    /// Called before attaching to a process.
    ///
    /// Allow Process plug-ins to execute some code before attaching a
    /// process.
    ///
    /// @return
    ///     Returns an error object.
    //------------------------------------------------------------------
    virtual Error
    WillAttachToProcessWithID (lldb::pid_t pid) 
    {
        return Error(); 
    }

    //------------------------------------------------------------------
    /// Called before attaching to a process.
    ///
    /// Allow Process plug-ins to execute some code before attaching a
    /// process.
    ///
    /// @return
    ///     Returns an error object.
    //------------------------------------------------------------------
    virtual Error
    WillAttachToProcessWithName (const char *process_name, bool wait_for_launch) 
    { 
        return Error(); 
    }

    //------------------------------------------------------------------
    /// Attach to a remote system via a URL
    ///
    /// @param[in] strm
    ///     A stream where output intended for the user 
    ///     (if the driver has a way to display that) generated during
    ///     the connection.  This may be NULL if no output is needed.A
    ///
    /// @param[in] remote_url
    ///     The URL format that we are connecting to.
    ///
    /// @return
    ///     Returns an error object.
    //------------------------------------------------------------------
    virtual Error
    DoConnectRemote (Stream *strm, const char *remote_url)
    {
        Error error;
        error.SetErrorString ("remote connections are not supported");
        return error;
    }

    //------------------------------------------------------------------
    /// Attach to an existing process using a process ID.
    ///
    /// @param[in] pid
    ///     The process ID that we should attempt to attach to.
    ///
    /// @return
    ///     Returns \a pid if attaching was successful, or
    ///     LLDB_INVALID_PROCESS_ID if attaching fails.
    //------------------------------------------------------------------
    virtual Error
    DoAttachToProcessWithID (lldb::pid_t pid)
    {
        Error error;
        error.SetErrorStringWithFormat("error: %s does not support attaching to a process by pid", GetShortPluginName());
        return error;
    }

    //------------------------------------------------------------------
    /// Attach to an existing process using a process ID.
    ///
    /// @param[in] pid
    ///     The process ID that we should attempt to attach to.
    ///
    /// @param[in] attach_info
    ///     Information on how to do the attach. For example, GetUserID()
    ///     will return the uid to attach as.
    ///
    /// @return
    ///     Returns \a pid if attaching was successful, or
    ///     LLDB_INVALID_PROCESS_ID if attaching fails.
    /// hanming : need flag
    //------------------------------------------------------------------
    virtual Error
    DoAttachToProcessWithID (lldb::pid_t pid,  const ProcessAttachInfo &attach_info)
    {
        Error error;
        error.SetErrorStringWithFormat("error: %s does not support attaching to a process by pid", GetShortPluginName());
        return error;
    }

    //------------------------------------------------------------------
    /// Attach to an existing process using a partial process name.
    ///
    /// @param[in] process_name
    ///     The name of the process to attach to.
    ///
    /// @param[in] wait_for_launch
    ///     If \b true, wait for the process to be launched and attach
    ///     as soon as possible after it does launch. If \b false, then
    ///     search for a matching process the currently exists.
    ///
    /// @param[in] attach_info
    ///     Information on how to do the attach. For example, GetUserID()
    ///     will return the uid to attach as.
    ///
    /// @return
    ///     Returns \a pid if attaching was successful, or
    ///     LLDB_INVALID_PROCESS_ID if attaching fails.
    //------------------------------------------------------------------
    virtual Error
    DoAttachToProcessWithName (const char *process_name, bool wait_for_launch, const ProcessAttachInfo &attach_info) 
    {
        Error error;
        error.SetErrorString("attach by name is not supported");
        return error;
    }

    //------------------------------------------------------------------
    /// Called after attaching a process.
    ///
    /// Allow Process plug-ins to execute some code after attaching to
    /// a process.
    //------------------------------------------------------------------
    virtual void
    DidAttach () {}


    //------------------------------------------------------------------
    /// Called before launching to a process.
    ///
    /// Allow Process plug-ins to execute some code before launching a
    /// process.
    ///
    /// @return
    ///     Returns an error object.
    //------------------------------------------------------------------
    virtual Error
    WillLaunch (Module* module)
    {
        return Error();
    }

    //------------------------------------------------------------------
    /// Launch a new process.
    ///
    /// Launch a new process by spawning a new process using \a module's
    /// file as the file to launch. Arguments are given in \a argv,
    /// and the environment variables are in \a envp. Standard input
    /// and output files can be optionally re-directed to \a stdin_path,
    /// \a stdout_path, and \a stderr_path.
    ///
    /// @param[in] module
    ///     The module from which to extract the file specification and
    ///     launch.
    ///
    /// @param[in] argv
    ///     The argument array.
    ///
    /// @param[in] envp
    ///     The environment array.
    ///
    /// @param[in] launch_flags
    ///     Flags to modify the launch (@see lldb::LaunchFlags)
    ///
    /// @param[in] stdin_path
    ///     The path to use when re-directing the STDIN of the new
    ///     process. If all stdXX_path arguments are NULL, a pseudo
    ///     terminal will be used.
    ///
    /// @param[in] stdout_path
    ///     The path to use when re-directing the STDOUT of the new
    ///     process. If all stdXX_path arguments are NULL, a pseudo
    ///     terminal will be used.
    ///
    /// @param[in] stderr_path
    ///     The path to use when re-directing the STDERR of the new
    ///     process. If all stdXX_path arguments are NULL, a pseudo
    ///     terminal will be used.
    ///
    /// @param[in] working_directory
    ///     The working directory to have the child process run in
    ///
    /// @return
    ///     A new valid process ID, or LLDB_INVALID_PROCESS_ID if
    ///     launching fails.
    //------------------------------------------------------------------
    virtual Error
    DoLaunch (Module *exe_module,
              const ProcessLaunchInfo &launch_info)
    {
        Error error;
        error.SetErrorStringWithFormat("error: %s does not support launching processes", GetShortPluginName());
        return error;
    }

    
    //------------------------------------------------------------------
    /// Called after launching a process.
    ///
    /// Allow Process plug-ins to execute some code after launching
    /// a process.
    //------------------------------------------------------------------
    virtual void
    DidLaunch () {}



    //------------------------------------------------------------------
    /// Called before resuming to a process.
    ///
    /// Allow Process plug-ins to execute some code before resuming a
    /// process.
    ///
    /// @return
    ///     Returns an error object.
    //------------------------------------------------------------------
    virtual Error
    WillResume () { return Error(); }

    //------------------------------------------------------------------
    /// Resumes all of a process's threads as configured using the
    /// Thread run control functions.
    ///
    /// Threads for a process should be updated with one of the run
    /// control actions (resume, step, or suspend) that they should take
    /// when the process is resumed. If no run control action is given
    /// to a thread it will be resumed by default.
    ///
    /// @return
    ///     Returns \b true if the process successfully resumes using
    ///     the thread run control actions, \b false otherwise.
    ///
    /// @see Thread:Resume()
    /// @see Thread:Step()
    /// @see Thread:Suspend()
    //------------------------------------------------------------------
    virtual Error
    DoResume ()
    {
        Error error;
        error.SetErrorStringWithFormat("error: %s does not support resuming processes", GetShortPluginName());
        return error;
    }


    //------------------------------------------------------------------
    /// Called after resuming a process.
    ///
    /// Allow Process plug-ins to execute some code after resuming
    /// a process.
    //------------------------------------------------------------------
    virtual void
    DidResume () {}


    //------------------------------------------------------------------
    /// Called before halting to a process.
    ///
    /// Allow Process plug-ins to execute some code before halting a
    /// process.
    ///
    /// @return
    ///     Returns an error object.
    //------------------------------------------------------------------
    virtual Error
    WillHalt () { return Error(); }

    //------------------------------------------------------------------
    /// Halts a running process.
    ///
    /// DoHalt must produce one and only one stop StateChanged event if it actually
    /// stops the process.  If the stop happens through some natural event (for
    /// instance a SIGSTOP), then forwarding that event will do.  Otherwise, you must 
    /// generate the event manually.  Note also, the private event thread is stopped when 
    /// DoHalt is run to prevent the events generated while halting to trigger
    /// other state changes before the halt is complete.
    ///
    /// @param[out] caused_stop
    ///     If true, then this Halt caused the stop, otherwise, the 
    ///     process was already stopped.
    ///
    /// @return
    ///     Returns \b true if the process successfully halts, \b false
    ///     otherwise.
    //------------------------------------------------------------------
    virtual Error
    DoHalt (bool &caused_stop)
    {
        Error error;
        error.SetErrorStringWithFormat("error: %s does not support halting processes", GetShortPluginName());
        return error;
    }


    //------------------------------------------------------------------
    /// Called after halting a process.
    ///
    /// Allow Process plug-ins to execute some code after halting
    /// a process.
    //------------------------------------------------------------------
    virtual void
    DidHalt () {}

    //------------------------------------------------------------------
    /// Called before detaching from a process.
    ///
    /// Allow Process plug-ins to execute some code before detaching
    /// from a process.
    ///
    /// @return
    ///     Returns an error object.
    //------------------------------------------------------------------
    virtual Error
    WillDetach () 
    {
        return Error(); 
    }

    //------------------------------------------------------------------
    /// Detaches from a running or stopped process.
    ///
    /// @return
    ///     Returns \b true if the process successfully detaches, \b
    ///     false otherwise.
    //------------------------------------------------------------------
    virtual Error
    DoDetach ()
    {
        Error error;
        error.SetErrorStringWithFormat("error: %s does not support detaching from processes", GetShortPluginName());
        return error;
    }


    //------------------------------------------------------------------
    /// Called after detaching from a process.
    ///
    /// Allow Process plug-ins to execute some code after detaching
    /// from a process.
    //------------------------------------------------------------------
    virtual void
    DidDetach () {}

    //------------------------------------------------------------------
    /// Called before sending a signal to a process.
    ///
    /// Allow Process plug-ins to execute some code before sending a
    /// signal to a process.
    ///
    /// @return
    ///     Returns no error if it is safe to proceed with a call to
    ///     Process::DoSignal(int), otherwise an error describing what
    ///     prevents the signal from being sent.
    //------------------------------------------------------------------
    virtual Error
    WillSignal () { return Error(); }

    //------------------------------------------------------------------
    /// Sends a process a UNIX signal \a signal.
    ///
    /// @return
    ///     Returns an error object.
    //------------------------------------------------------------------
    virtual Error
    DoSignal (int signal)
    {
        Error error;
        error.SetErrorStringWithFormat("error: %s does not support senging signals to processes", GetShortPluginName());
        return error;
    }

    virtual Error
    WillDestroy () { return Error(); }

    virtual Error
    DoDestroy () = 0;

    virtual void
    DidDestroy () { }


    //------------------------------------------------------------------
    /// Called after sending a signal to a process.
    ///
    /// Allow Process plug-ins to execute some code after sending a
    /// signal to a process.
    //------------------------------------------------------------------
    virtual void
    DidSignal () {}

    //------------------------------------------------------------------
    /// Currently called as part of ShouldStop.
    /// FIXME: Should really happen when the target stops before the
    /// event is taken from the queue...
    ///
    /// This callback is called as the event
    /// is about to be queued up to allow Process plug-ins to execute
    /// some code prior to clients being notified that a process was
    /// stopped. Common operations include updating the thread list,
    /// invalidating any thread state (registers, stack, etc) prior to
    /// letting the notification go out.
    ///
    //------------------------------------------------------------------
    virtual void
    RefreshStateAfterStop () = 0;

    //------------------------------------------------------------------
    /// Get the target object pointer for this module.
    ///
    /// @return
    ///     A Target object pointer to the target that owns this
    ///     module.
    //------------------------------------------------------------------
    Target &
    GetTarget ()
    {
        return m_target;
    }

    //------------------------------------------------------------------
    /// Get the const target object pointer for this module.
    ///
    /// @return
    ///     A const Target object pointer to the target that owns this
    ///     module.
    //------------------------------------------------------------------
    const Target &
    GetTarget () const
    {
        return m_target;
    }

    //------------------------------------------------------------------
    /// Flush all data in the process.
    ///
    /// Flush the memory caches, all threads, and any other cached data
    /// in the process.
    ///
    /// This function can be called after a world changing event like
    /// adding a new symbol file, or after the process makes a large
    /// context switch (from boot ROM to booted into an OS).
    //------------------------------------------------------------------
    void
    Flush ();

    //------------------------------------------------------------------
    /// Get accessor for the current process state.
    ///
    /// @return
    ///     The current state of the process.
    ///
    /// @see lldb::StateType
    //------------------------------------------------------------------
    lldb::StateType
    GetState ();
    
    ExecutionResults
    RunThreadPlan (ExecutionContext &exe_ctx,    
                    lldb::ThreadPlanSP &thread_plan_sp,
                    bool stop_others,
                    bool run_others,
                    bool discard_on_error,
                    uint32_t timeout_usec,
                    Stream &errors);

    static const char *
    ExecutionResultAsCString (ExecutionResults result);

    void
    GetStatus (Stream &ostrm);

    size_t
    GetThreadStatus (Stream &ostrm, 
                     bool only_threads_with_stop_reason,
                     uint32_t start_frame, 
                     uint32_t num_frames, 
                     uint32_t num_frames_with_source);

    void
    SendAsyncInterrupt ();
    
protected:
    
    void
    SetState (lldb::EventSP &event_sp);

    lldb::StateType
    GetPrivateState ();

    //------------------------------------------------------------------
    /// The "private" side of resuming a process.  This doesn't alter the
    /// state of m_run_lock, but just causes the process to resume.
    ///
    /// @return
    ///     An Error object describing the success or failure of the resume.
    //------------------------------------------------------------------
    Error
    PrivateResume ();

    //------------------------------------------------------------------
    // Called internally
    //------------------------------------------------------------------
    void
    CompleteAttach ();
    
public:
    //------------------------------------------------------------------
    /// Get the exit status for a process.
    ///
    /// @return
    ///     The process's return code, or -1 if the current process
    ///     state is not eStateExited.
    //------------------------------------------------------------------
    int
    GetExitStatus ();

    //------------------------------------------------------------------
    /// Get a textual description of what the process exited.
    ///
    /// @return
    ///     The textual description of why the process exited, or NULL
    ///     if there is no description available.
    //------------------------------------------------------------------
    const char *
    GetExitDescription ();


    virtual void
    DidExit ()
    {
    }

    //------------------------------------------------------------------
    /// Get the Modification ID of the process.
    ///
    /// @return
    ///     The modification ID of the process.
    //------------------------------------------------------------------
    ProcessModID
    GetModID () const
    {
        return m_mod_id;
    }
    
    const ProcessModID &
    GetModIDRef () const
    {
        return m_mod_id;
    }
    
    uint32_t
    GetStopID () const
    {
        return m_mod_id.GetStopID();
    }
    
    uint32_t
    GetResumeID () const
    {
        return m_mod_id.GetResumeID();
    }
    
    uint32_t
    GetLastUserExpressionResumeID () const
    {
        return m_mod_id.GetLastUserExpressionResumeID();
    }
    
    //------------------------------------------------------------------
    /// Set accessor for the process exit status (return code).
    ///
    /// Sometimes a child exits and the exit can be detected by global
    /// functions (signal handler for SIGCHLD for example). This
    /// accessor allows the exit status to be set from an external
    /// source.
    ///
    /// Setting this will cause a eStateExited event to be posted to
    /// the process event queue.
    ///
    /// @param[in] exit_status
    ///     The value for the process's return code.
    ///
    /// @see lldb::StateType
    //------------------------------------------------------------------
    virtual bool
    SetExitStatus (int exit_status, const char *cstr);

    //------------------------------------------------------------------
    /// Check if a process is still alive.
    ///
    /// @return
    ///     Returns \b true if the process is still valid, \b false
    ///     otherwise.
    //------------------------------------------------------------------
    virtual bool
    IsAlive () = 0;

    //------------------------------------------------------------------
    /// Actually do the reading of memory from a process.
    ///
    /// Subclasses must override this function and can return fewer 
    /// bytes than requested when memory requests are too large. This
    /// class will break up the memory requests and keep advancing the
    /// arguments along as needed. 
    ///
    /// @param[in] vm_addr
    ///     A virtual load address that indicates where to start reading
    ///     memory from.
    ///
    /// @param[in] size
    ///     The number of bytes to read.
    ///
    /// @param[out] buf
    ///     A byte buffer that is at least \a size bytes long that
    ///     will receive the memory bytes.
    ///
    /// @return
    ///     The number of bytes that were actually read into \a buf.
    //------------------------------------------------------------------
    virtual size_t
    DoReadMemory (lldb::addr_t vm_addr, 
                  void *buf, 
                  size_t size,
                  Error &error) = 0;

    //------------------------------------------------------------------
    /// Read of memory from a process.
    ///
    /// This function will read memory from the current process's
    /// address space and remove any traps that may have been inserted
    /// into the memory.
    ///
    /// This function is not meant to be overridden by Process
    /// subclasses, the subclasses should implement
    /// Process::DoReadMemory (lldb::addr_t, size_t, void *).
    ///
    /// @param[in] vm_addr
    ///     A virtual load address that indicates where to start reading
    ///     memory from.
    ///
    /// @param[out] buf
    ///     A byte buffer that is at least \a size bytes long that
    ///     will receive the memory bytes.
    ///
    /// @param[in] size
    ///     The number of bytes to read.
    ///
    /// @return
    ///     The number of bytes that were actually read into \a buf. If
    ///     the returned number is greater than zero, yet less than \a
    ///     size, then this function will get called again with \a 
    ///     vm_addr, \a buf, and \a size updated appropriately. Zero is
    ///     returned to indicate an error.
    //------------------------------------------------------------------
    virtual size_t
    ReadMemory (lldb::addr_t vm_addr, 
                void *buf, 
                size_t size,
                Error &error);

    //------------------------------------------------------------------
    /// Read a NULL terminated C string from memory
    ///
    /// This function will read a cache page at a time until the NULL
    /// C string terminator is found. It will stop reading if the NULL
    /// termination byte isn't found before reading \a cstr_max_len
    /// bytes, and the results are always guaranteed to be NULL 
    /// terminated (at most cstr_max_len - 1 bytes will be read).
    //------------------------------------------------------------------
    size_t
    ReadCStringFromMemory (lldb::addr_t vm_addr, 
                           char *cstr, 
                           size_t cstr_max_len,
                           Error &error);

    size_t
    ReadCStringFromMemory (lldb::addr_t vm_addr,
                           std::string &out_str,
                           Error &error);

    size_t
    ReadMemoryFromInferior (lldb::addr_t vm_addr, 
                            void *buf, 
                            size_t size,
                            Error &error);
    
    //------------------------------------------------------------------
    /// Reads an unsigned integer of the specified byte size from 
    /// process memory.
    ///
    /// @param[in] load_addr
    ///     A load address of the integer to read.
    ///
    /// @param[in] byte_size
    ///     The size in byte of the integer to read.
    ///
    /// @param[in] fail_value
    ///     The value to return if we fail to read an integer.
    ///
    /// @param[out] error
    ///     An error that indicates the success or failure of this
    ///     operation. If error indicates success (error.Success()), 
    ///     then the value returned can be trusted, otherwise zero
    ///     will be returned.
    ///
    /// @return
    ///     The unsigned integer that was read from the process memory
    ///     space. If the integer was smaller than a uint64_t, any
    ///     unused upper bytes will be zero filled. If the process
    ///     byte order differs from the host byte order, the integer
    ///     value will be appropriately byte swapped into host byte
    ///     order.
    //------------------------------------------------------------------
    uint64_t
    ReadUnsignedIntegerFromMemory (lldb::addr_t load_addr, 
                                   size_t byte_size,
                                   uint64_t fail_value, 
                                   Error &error);
    
    lldb::addr_t
    ReadPointerFromMemory (lldb::addr_t vm_addr, 
                           Error &error);

    bool
    WritePointerToMemory (lldb::addr_t vm_addr, 
                          lldb::addr_t ptr_value, 
                          Error &error);

    //------------------------------------------------------------------
    /// Actually do the writing of memory to a process.
    ///
    /// @param[in] vm_addr
    ///     A virtual load address that indicates where to start writing
    ///     memory to.
    ///
    /// @param[in] buf
    ///     A byte buffer that is at least \a size bytes long that
    ///     contains the data to write.
    ///
    /// @param[in] size
    ///     The number of bytes to write.
    ///
    /// @param[out] error
    ///     An error value in case the memory write fails.
    ///
    /// @return
    ///     The number of bytes that were actually written.
    //------------------------------------------------------------------
    virtual size_t
    DoWriteMemory (lldb::addr_t vm_addr, const void *buf, size_t size, Error &error)
    {
        error.SetErrorStringWithFormat("error: %s does not support writing to processes", GetShortPluginName());
        return 0;
    }


    //------------------------------------------------------------------
    /// Write all or part of a scalar value to memory.
    ///
    /// The value contained in \a scalar will be swapped to match the
    /// byte order of the process that is being debugged. If \a size is
    /// less than the size of scalar, the least significate \a size bytes
    /// from scalar will be written. If \a size is larger than the byte
    /// size of scalar, then the extra space will be padded with zeros
    /// and the scalar value will be placed in the least significant
    /// bytes in memory.
    ///
    /// @param[in] vm_addr
    ///     A virtual load address that indicates where to start writing
    ///     memory to.
    ///
    /// @param[in] scalar
    ///     The scalar to write to the debugged process.
    ///
    /// @param[in] size
    ///     This value can be smaller or larger than the scalar value
    ///     itself. If \a size is smaller than the size of \a scalar, 
    ///     the least significant bytes in \a scalar will be used. If
    ///     \a size is larger than the byte size of \a scalar, then 
    ///     the extra space will be padded with zeros. If \a size is
    ///     set to UINT32_MAX, then the size of \a scalar will be used.
    ///
    /// @param[out] error
    ///     An error value in case the memory write fails.
    ///
    /// @return
    ///     The number of bytes that were actually written.
    //------------------------------------------------------------------
    size_t
    WriteScalarToMemory (lldb::addr_t vm_addr, 
                         const Scalar &scalar, 
                         uint32_t size, 
                         Error &error);

    size_t
    ReadScalarIntegerFromMemory (lldb::addr_t addr, 
                                 uint32_t byte_size, 
                                 bool is_signed, 
                                 Scalar &scalar, 
                                 Error &error);

    //------------------------------------------------------------------
    /// Write memory to a process.
    ///
    /// This function will write memory to the current process's
    /// address space and maintain any traps that might be present due
    /// to software breakpoints.
    ///
    /// This function is not meant to be overridden by Process
    /// subclasses, the subclasses should implement
    /// Process::DoWriteMemory (lldb::addr_t, size_t, void *).
    ///
    /// @param[in] vm_addr
    ///     A virtual load address that indicates where to start writing
    ///     memory to.
    ///
    /// @param[in] buf
    ///     A byte buffer that is at least \a size bytes long that
    ///     contains the data to write.
    ///
    /// @param[in] size
    ///     The number of bytes to write.
    ///
    /// @return
    ///     The number of bytes that were actually written.
    //------------------------------------------------------------------
    size_t
    WriteMemory (lldb::addr_t vm_addr, const void *buf, size_t size, Error &error);


    //------------------------------------------------------------------
    /// Actually allocate memory in the process.
    ///
    /// This function will allocate memory in the process's address
    /// space.  This can't rely on the generic function calling mechanism,
    /// since that requires this function.
    ///
    /// @param[in] size
    ///     The size of the allocation requested.
    ///
    /// @return
    ///     The address of the allocated buffer in the process, or
    ///     LLDB_INVALID_ADDRESS if the allocation failed.
    //------------------------------------------------------------------

    virtual lldb::addr_t
    DoAllocateMemory (size_t size, uint32_t permissions, Error &error)
    {
        error.SetErrorStringWithFormat("error: %s does not support allocating in the debug process", GetShortPluginName());
        return LLDB_INVALID_ADDRESS;
    }


    //------------------------------------------------------------------
    /// The public interface to allocating memory in the process.
    ///
    /// This function will allocate memory in the process's address
    /// space.  This can't rely on the generic function calling mechanism,
    /// since that requires this function.
    ///
    /// @param[in] size
    ///     The size of the allocation requested.
    ///
    /// @param[in] permissions
    ///     Or together any of the lldb::Permissions bits.  The permissions on
    ///     a given memory allocation can't be changed after allocation.  Note
    ///     that a block that isn't set writable can still be written on from lldb,
    ///     just not by the process itself.
    ///
    /// @param[in/out] error
    ///     An error object to fill in if things go wrong.
    /// @return
    ///     The address of the allocated buffer in the process, or
    ///     LLDB_INVALID_ADDRESS if the allocation failed.
    //------------------------------------------------------------------

    lldb::addr_t
    AllocateMemory (size_t size, uint32_t permissions, Error &error);

    virtual Error
    GetMemoryRegionInfo (lldb::addr_t load_addr, 
                        MemoryRegionInfo &range_info)
    {
        Error error;
        error.SetErrorString ("Process::GetMemoryRegionInfo() not supported");
        return error;
    }

    virtual Error
    GetWatchpointSupportInfo (uint32_t &num)
    {
        Error error;
        error.SetErrorString ("Process::GetWatchpointSupportInfo() not supported");
        return error;
    }

    virtual Error
    GetWatchpointSupportInfo (uint32_t &num, bool& after)
    {
        Error error;
        error.SetErrorString ("Process::GetWatchpointSupportInfo() not supported");
        return error;
    }
    
    lldb::ModuleSP
    ReadModuleFromMemory (const FileSpec& file_spec, 
                          lldb::addr_t header_addr,
                          bool add_image_to_target,
                          bool load_sections_in_target);

    //------------------------------------------------------------------
    /// Attempt to get the attributes for a region of memory in the process.
    ///
    /// It may be possible for the remote debug server to inspect attributes
    /// for a region of memory in the process, such as whether there is a
    /// valid page of memory at a given address or whether that page is 
    /// readable/writable/executable by the process.
    ///
    /// @param[in] load_addr
    ///     The address of interest in the process.
    ///
    /// @param[out] permissions
    ///     If this call returns successfully, this bitmask will have
    ///     its Permissions bits set to indicate whether the region is
    ///     readable/writable/executable.  If this call fails, the
    ///     bitmask values are undefined.
    ///
    /// @return
    ///     Returns true if it was able to determine the attributes of the
    ///     memory region.  False if not.
    //------------------------------------------------------------------

    virtual bool
    GetLoadAddressPermissions (lldb::addr_t load_addr, uint32_t &permissions)
    {
        MemoryRegionInfo range_info;
        permissions = 0;
        Error error (GetMemoryRegionInfo (load_addr, range_info));
        if (!error.Success())
            return false;
        if (range_info.GetReadable() == MemoryRegionInfo::eDontKnow 
            || range_info.GetWritable() == MemoryRegionInfo::eDontKnow 
            || range_info.GetExecutable() == MemoryRegionInfo::eDontKnow)
        {
            return false;
        }

        if (range_info.GetReadable() == MemoryRegionInfo::eYes)
            permissions |= lldb::ePermissionsReadable;

        if (range_info.GetWritable() == MemoryRegionInfo::eYes)
            permissions |= lldb::ePermissionsWritable;

        if (range_info.GetExecutable() == MemoryRegionInfo::eYes)
            permissions |= lldb::ePermissionsExecutable;

        return true;
    }

    //------------------------------------------------------------------
    /// Determines whether executing JIT-compiled code in this process 
    /// is possible.
    ///
    /// @return
    ///     True if execution of JIT code is possible; false otherwise.
    //------------------------------------------------------------------    
    bool CanJIT ();
    
    //------------------------------------------------------------------
    /// Sets whether executing JIT-compiled code in this process 
    /// is possible.
    ///
    /// @param[in] can_jit
    ///     True if execution of JIT code is possible; false otherwise.
    //------------------------------------------------------------------
    void SetCanJIT (bool can_jit);
    
    //------------------------------------------------------------------
    /// Actually deallocate memory in the process.
    ///
    /// This function will deallocate memory in the process's address
    /// space that was allocated with AllocateMemory.
    ///
    /// @param[in] ptr
    ///     A return value from AllocateMemory, pointing to the memory you
    ///     want to deallocate.
    ///
    /// @return
    ///     \btrue if the memory was deallocated, \bfalse otherwise.
    //------------------------------------------------------------------

    virtual Error
    DoDeallocateMemory (lldb::addr_t ptr)
    {
        Error error;
        error.SetErrorStringWithFormat("error: %s does not support deallocating in the debug process", GetShortPluginName());
        return error;
    }


    //------------------------------------------------------------------
    /// The public interface to deallocating memory in the process.
    ///
    /// This function will deallocate memory in the process's address
    /// space that was allocated with AllocateMemory.
    ///
    /// @param[in] ptr
    ///     A return value from AllocateMemory, pointing to the memory you
    ///     want to deallocate.
    ///
    /// @return
    ///     \btrue if the memory was deallocated, \bfalse otherwise.
    //------------------------------------------------------------------

    Error
    DeallocateMemory (lldb::addr_t ptr);

    //------------------------------------------------------------------
    /// Get any available STDOUT.
    ///
    /// If the process was launched without supplying valid file paths
    /// for stdin, stdout, and stderr, then the Process class might
    /// try to cache the STDOUT for the process if it is able. Events
    /// will be queued indicating that there is STDOUT available that
    /// can be retrieved using this function.
    ///
    /// @param[out] buf
    ///     A buffer that will receive any STDOUT bytes that are
    ///     currently available.
    ///
    /// @param[out] buf_size
    ///     The size in bytes for the buffer \a buf.
    ///
    /// @return
    ///     The number of bytes written into \a buf. If this value is
    ///     equal to \a buf_size, another call to this function should
    ///     be made to retrieve more STDOUT data.
    //------------------------------------------------------------------
    virtual size_t
    GetSTDOUT (char *buf, size_t buf_size, Error &error);

    //------------------------------------------------------------------
    /// Get any available STDERR.
    ///
    /// If the process was launched without supplying valid file paths
    /// for stdin, stdout, and stderr, then the Process class might
    /// try to cache the STDERR for the process if it is able. Events
    /// will be queued indicating that there is STDERR available that
    /// can be retrieved using this function.
    ///
    /// @param[out] buf
    ///     A buffer that will receive any STDERR bytes that are
    ///     currently available.
    ///
    /// @param[out] buf_size
    ///     The size in bytes for the buffer \a buf.
    ///
    /// @return
    ///     The number of bytes written into \a buf. If this value is
    ///     equal to \a buf_size, another call to this function should
    ///     be made to retrieve more STDERR data.
    //------------------------------------------------------------------
    virtual size_t
    GetSTDERR (char *buf, size_t buf_size, Error &error);

    virtual size_t
    PutSTDIN (const char *buf, size_t buf_size, Error &error) 
    {
        error.SetErrorString("stdin unsupported");
        return 0;
    }

    //----------------------------------------------------------------------
    // Process Breakpoints
    //----------------------------------------------------------------------
    size_t
    GetSoftwareBreakpointTrapOpcode (BreakpointSite* bp_site);

    virtual Error
    EnableBreakpoint (BreakpointSite *bp_site)
    {
        Error error;
        error.SetErrorStringWithFormat("error: %s does not support enabling breakpoints", GetShortPluginName());
        return error;
    }


    virtual Error
    DisableBreakpoint (BreakpointSite *bp_site)
    {
        Error error;
        error.SetErrorStringWithFormat("error: %s does not support disabling breakpoints", GetShortPluginName());
        return error;
    }


    // This is implemented completely using the lldb::Process API. Subclasses
    // don't need to implement this function unless the standard flow of
    // read existing opcode, write breakpoint opcode, verify breakpoint opcode
    // doesn't work for a specific process plug-in.
    virtual Error
    EnableSoftwareBreakpoint (BreakpointSite *bp_site);

    // This is implemented completely using the lldb::Process API. Subclasses
    // don't need to implement this function unless the standard flow of
    // restoring original opcode in memory and verifying the restored opcode
    // doesn't work for a specific process plug-in.
    virtual Error
    DisableSoftwareBreakpoint (BreakpointSite *bp_site);

    BreakpointSiteList &
    GetBreakpointSiteList();

    const BreakpointSiteList &
    GetBreakpointSiteList() const;

    void
    DisableAllBreakpointSites ();

    Error
    ClearBreakpointSiteByID (lldb::user_id_t break_id);

    lldb::break_id_t
    CreateBreakpointSite (const lldb::BreakpointLocationSP &owner,
                          bool use_hardware);

    Error
    DisableBreakpointSiteByID (lldb::user_id_t break_id);

    Error
    EnableBreakpointSiteByID (lldb::user_id_t break_id);


    // BreakpointLocations use RemoveOwnerFromBreakpointSite to remove
    // themselves from the owner's list of this breakpoint sites.  This has to
    // be a static function because you can't be sure that removing the
    // breakpoint from it's containing map won't delete the breakpoint site,
    // and doing that in an instance method isn't copasetic.
    void
    RemoveOwnerFromBreakpointSite (lldb::user_id_t owner_id,
                                   lldb::user_id_t owner_loc_id,
                                   lldb::BreakpointSiteSP &bp_site_sp);

    //----------------------------------------------------------------------
    // Process Watchpoints (optional)
    //----------------------------------------------------------------------
    virtual Error
    EnableWatchpoint (Watchpoint *wp);

    virtual Error
    DisableWatchpoint (Watchpoint *wp);

    //------------------------------------------------------------------
    // Thread Queries
    //------------------------------------------------------------------
    virtual bool
    UpdateThreadList (ThreadList &old_thread_list, ThreadList &new_thread_list) = 0;

    void
    UpdateThreadListIfNeeded ();

    ThreadList &
    GetThreadList ()
    {
        return m_thread_list;
    }


    uint32_t
    GetNextThreadIndexID ();

    //------------------------------------------------------------------
    // Event Handling
    //------------------------------------------------------------------
    lldb::StateType
    GetNextEvent (lldb::EventSP &event_sp);

    lldb::StateType
    WaitForProcessToStop (const TimeValue *timeout, lldb::EventSP *event_sp_ptr = NULL);

    lldb::StateType
    WaitForStateChangedEvents (const TimeValue *timeout, lldb::EventSP &event_sp);
    
    Event *
    PeekAtStateChangedEvents ();
    

    class
    ProcessEventHijacker
    {
    public:
        ProcessEventHijacker (Process &process, Listener *listener) :
            m_process (process),
            m_listener (listener)
        {
            m_process.HijackProcessEvents (listener);
        }
        ~ProcessEventHijacker ()
        {
            m_process.RestoreProcessEvents();
        }
         
    private:
        Process &m_process;
        Listener *m_listener;
    };
    friend class ProcessEventHijacker;
    //------------------------------------------------------------------
    /// If you need to ensure that you and only you will hear about some public
    /// event, then make a new listener, set to listen to process events, and
    /// then call this with that listener.  Then you will have to wait on that
    /// listener explicitly for events (rather than using the GetNextEvent & WaitFor*
    /// calls above.  Be sure to call RestoreProcessEvents when you are done.
    ///
    /// @param[in] listener
    ///     This is the new listener to whom all process events will be delivered.
    ///
    /// @return
    ///     Returns \b true if the new listener could be installed,
    ///     \b false otherwise.
    //------------------------------------------------------------------
    bool
    HijackProcessEvents (Listener *listener);
    
    //------------------------------------------------------------------
    /// Restores the process event broadcasting to its normal state.
    ///
    //------------------------------------------------------------------
    void
    RestoreProcessEvents ();

protected:
    //------------------------------------------------------------------
    /// This is the part of the event handling that for a process event.
    /// It decides what to do with the event and returns true if the
    /// event needs to be propagated to the user, and false otherwise.
    /// If the event is not propagated, this call will most likely set
    /// the target to executing again.
    ///
    /// @param[in] event_ptr
    ///     This is the event we are handling.
    ///
    /// @return
    ///     Returns \b true if the event should be reported to the
    ///     user, \b false otherwise.
    //------------------------------------------------------------------
    bool
    ShouldBroadcastEvent (Event *event_ptr);

public:
    const lldb::ABISP &
    GetABI ();

    OperatingSystem *
    GetOperatingSystem ()
    {
        return m_os_ap.get();
    }
    

    virtual LanguageRuntime *
    GetLanguageRuntime (lldb::LanguageType language, bool retry_if_null = true);

    virtual CPPLanguageRuntime *
    GetCPPLanguageRuntime (bool retry_if_null = true);

    virtual ObjCLanguageRuntime *
    GetObjCLanguageRuntime (bool retry_if_null = true);
    
    bool
    IsPossibleDynamicValue (ValueObject& in_value);
    
    bool
    IsRunning () const;
    
    DynamicCheckerFunctions *GetDynamicCheckers()
    {
        return m_dynamic_checkers_ap.get();
    }
    
    void SetDynamicCheckers(DynamicCheckerFunctions *dynamic_checkers)
    {
        m_dynamic_checkers_ap.reset(dynamic_checkers);
    }

    //------------------------------------------------------------------
    /// Call this to set the lldb in the mode where it breaks on new thread
    /// creations, and then auto-restarts.  This is useful when you are trying
    /// to run only one thread, but either that thread or the kernel is creating
    /// new threads in the process.  If you stop when the thread is created, you
    /// can immediately suspend it, and keep executing only the one thread you intend.
    ///
    /// @return
    ///     Returns \b true if we were able to start up the notification
    ///     \b false otherwise.
    //------------------------------------------------------------------
    virtual bool
    StartNoticingNewThreads()
    {   
        return true;
    }
    
    //------------------------------------------------------------------
    /// Call this to turn off the stop & notice new threads mode.
    ///
    /// @return
    ///     Returns \b true if we were able to start up the notification
    ///     \b false otherwise.
    //------------------------------------------------------------------
    virtual bool
    StopNoticingNewThreads()
    {   
        return true;
    }
    
    void
    SetRunningUserExpression (bool on);
    
    //------------------------------------------------------------------
    // lldb::ExecutionContextScope pure virtual functions
    //------------------------------------------------------------------
    virtual lldb::TargetSP
    CalculateTarget ();
    
    virtual lldb::ProcessSP
    CalculateProcess ()
    {
        return shared_from_this();
    }
    
    virtual lldb::ThreadSP
    CalculateThread ()
    {
        return lldb::ThreadSP();
    }
    
    virtual lldb::StackFrameSP
    CalculateStackFrame ()
    {
        return lldb::StackFrameSP();
    }

    virtual void
    CalculateExecutionContext (ExecutionContext &exe_ctx);
    
    void
    SetSTDIOFileDescriptor (int file_descriptor);

    //------------------------------------------------------------------
    // Add a permanent region of memory that should never be read or 
    // written to. This can be used to ensure that memory reads or writes
    // to certain areas of memory never end up being sent to the 
    // DoReadMemory or DoWriteMemory functions which can improve 
    // performance.
    //------------------------------------------------------------------
    void
    AddInvalidMemoryRegion (const LoadRange &region);
    
    //------------------------------------------------------------------
    // Remove a permanent region of memory that should never be read or 
    // written to that was previously added with AddInvalidMemoryRegion.
    //------------------------------------------------------------------
    bool
    RemoveInvalidMemoryRange (const LoadRange &region);
    
    //------------------------------------------------------------------
    // If the setup code of a thread plan needs to do work that might involve 
    // calling a function in the target, it should not do that work directly
    // in one of the thread plan functions (DidPush/WillResume) because
    // such work needs to be handled carefully.  Instead, put that work in
    // a PreResumeAction callback, and register it with the process.  It will
    // get done before the actual "DoResume" gets called.
    //------------------------------------------------------------------
    
    typedef bool (PreResumeActionCallback)(void *);

    void
    AddPreResumeAction (PreResumeActionCallback callback, void *baton);
    
    bool
    RunPreResumeActions ();
    
    void
    ClearPreResumeActions ();
                              
    ReadWriteLock &
    GetRunLock ()
    {
        return m_run_lock;
    }

protected:
    //------------------------------------------------------------------
    // NextEventAction provides a way to register an action on the next
    // event that is delivered to this process.  There is currently only
    // one next event action allowed in the process at one time.  If a
    // new "NextEventAction" is added while one is already present, the
    // old action will be discarded (with HandleBeingUnshipped called 
    // after it is discarded.)
    //------------------------------------------------------------------
    class NextEventAction
    {
    public:
        typedef enum EventActionResult
        {
            eEventActionSuccess,
            eEventActionRetry,
            eEventActionExit
        } EventActionResult;
        
        NextEventAction (Process *process) : 
            m_process(process)
        {
        }

        virtual
        ~NextEventAction() 
        {
        }
        
        virtual EventActionResult PerformAction (lldb::EventSP &event_sp) = 0;
        virtual void HandleBeingUnshipped () {}
        virtual EventActionResult HandleBeingInterrupted () = 0;
        virtual const char *GetExitString() = 0;
    protected:
        Process *m_process;
    };
    
    void SetNextEventAction (Process::NextEventAction *next_event_action)
    {
        if (m_next_event_action_ap.get())
            m_next_event_action_ap->HandleBeingUnshipped();

        m_next_event_action_ap.reset(next_event_action);
    }
    
    // This is the completer for Attaching:
    class AttachCompletionHandler : public NextEventAction
    {
    public:
        AttachCompletionHandler (Process *process, uint32_t exec_count) :
            NextEventAction (process),
            m_exec_count (exec_count)
        {
        }

        virtual 
        ~AttachCompletionHandler() 
        {
        }
        
        virtual EventActionResult PerformAction (lldb::EventSP &event_sp);
        virtual EventActionResult HandleBeingInterrupted ();
        virtual const char *GetExitString();
    private:
        uint32_t m_exec_count;
        std::string m_exit_string;
    };

    bool 
    HijackPrivateProcessEvents (Listener *listener);
    
    void 
    RestorePrivateProcessEvents ();
    
    bool
    PrivateStateThreadIsValid () const
    {
        return IS_VALID_LLDB_HOST_THREAD(m_private_state_thread);
    }

    //------------------------------------------------------------------
    // Type definitions
    //------------------------------------------------------------------
    typedef std::map<lldb::LanguageType, lldb::LanguageRuntimeSP> LanguageRuntimeCollection;

    struct PreResumeCallbackAndBaton
    {
        bool (*callback) (void *);
        void *baton;
        PreResumeCallbackAndBaton (PreResumeActionCallback in_callback, void *in_baton) :
            callback (in_callback),
            baton (in_baton)
        {
        }
    };
    
    //------------------------------------------------------------------
    // Member variables
    //------------------------------------------------------------------
    Target &                    m_target;               ///< The target that owns this process.
    ThreadSafeValue<lldb::StateType>  m_public_state;
    ThreadSafeValue<lldb::StateType>  m_private_state; // The actual state of our process
    Broadcaster                 m_private_state_broadcaster;  // This broadcaster feeds state changed events into the private state thread's listener.
    Broadcaster                 m_private_state_control_broadcaster; // This is the control broadcaster, used to pause, resume & stop the private state thread.
    Listener                    m_private_state_listener;     // This is the listener for the private state thread.
    Predicate<bool>             m_private_state_control_wait; /// This Predicate is used to signal that a control operation is complete.
    lldb::thread_t              m_private_state_thread;  // Thread ID for the thread that watches interal state events
    ProcessModID                m_mod_id;               ///< Tracks the state of the process over stops and other alterations.
    uint32_t                    m_thread_index_id;      ///< Each thread is created with a 1 based index that won't get re-used.
    int                         m_exit_status;          ///< The exit status of the process, or -1 if not set.
    std::string                 m_exit_string;          ///< A textual description of why a process exited.
    ThreadList                  m_thread_list;          ///< The threads for this process.
    std::vector<Notifications>  m_notifications;        ///< The list of notifications that this process can deliver.
    std::vector<lldb::addr_t>   m_image_tokens;
    Listener                    &m_listener;
    BreakpointSiteList          m_breakpoint_site_list; ///< This is the list of breakpoint locations we intend to insert in the target.
    std::auto_ptr<DynamicLoader> m_dyld_ap;
    std::auto_ptr<DynamicCheckerFunctions> m_dynamic_checkers_ap; ///< The functions used by the expression parser to validate data that expressions use.
    std::auto_ptr<OperatingSystem> m_os_ap;
    UnixSignals                 m_unix_signals;         /// This is the current signal set for this process.
    lldb::ABISP                 m_abi_sp;
    lldb::InputReaderSP         m_process_input_reader;
    Communication 				m_stdio_communication;
    Mutex        				m_stdio_communication_mutex;
    std::string                 m_stdout_data;
    std::string                 m_stderr_data;
    MemoryCache                 m_memory_cache;
    AllocatedMemoryCache        m_allocated_memory_cache;
    bool                        m_should_detach;   /// Should we detach if the process object goes away with an explicit call to Kill or Detach?
    LanguageRuntimeCollection 	m_language_runtimes;
    std::auto_ptr<NextEventAction> m_next_event_action_ap;
    std::vector<PreResumeCallbackAndBaton> m_pre_resume_actions;
    ReadWriteLock               m_run_lock;
    Predicate<bool>             m_currently_handling_event;
    bool                        m_finalize_called;

    enum {
        eCanJITDontKnow= 0,
        eCanJITYes,
        eCanJITNo
    } m_can_jit;

    size_t
    RemoveBreakpointOpcodesFromBuffer (lldb::addr_t addr, size_t size, uint8_t *buf) const;

    void
    SynchronouslyNotifyStateChanged (lldb::StateType state);

    void
    SetPublicState (lldb::StateType new_state);

    void
    SetPrivateState (lldb::StateType state);

    bool
    StartPrivateStateThread (bool force = false);

    void
    StopPrivateStateThread ();

    void
    PausePrivateStateThread ();

    void
    ResumePrivateStateThread ();

    static void *
    PrivateStateThread (void *arg);

    void *
    RunPrivateStateThread ();

    void
    HandlePrivateEvent (lldb::EventSP &event_sp);

    lldb::StateType
    WaitForProcessStopPrivate (const TimeValue *timeout, lldb::EventSP &event_sp);

    // This waits for both the state change broadcaster, and the control broadcaster.
    // If control_only, it only waits for the control broadcaster.

    bool
    WaitForEventsPrivate (const TimeValue *timeout, lldb::EventSP &event_sp, bool control_only);

    lldb::StateType
    WaitForStateChangedEventsPrivate (const TimeValue *timeout, lldb::EventSP &event_sp);

    lldb::StateType
    WaitForState (const TimeValue *timeout,
                  const lldb::StateType *match_states,
                  const uint32_t num_match_states);

    size_t
    WriteMemoryPrivate (lldb::addr_t addr, const void *buf, size_t size, Error &error);
    
    void
    AppendSTDOUT (const char *s, size_t len);
    
    void
    AppendSTDERR (const char *s, size_t len);
    
    static void
    STDIOReadThreadBytesReceived (void *baton, const void *src, size_t src_len);
    
    void
    PushProcessInputReader ();
    
    void 
    PopProcessInputReader ();
    
    void
    ResetProcessInputReader ();
    
    static size_t
    ProcessInputReaderCallback (void *baton,
                                InputReader &reader,
                                lldb::InputReaderAction notification,
                                const char *bytes,
                                size_t bytes_len);
    
    
private:
    //------------------------------------------------------------------
    // For Process only
    //------------------------------------------------------------------
    void ControlPrivateStateThread (uint32_t signal);

    DISALLOW_COPY_AND_ASSIGN (Process);

};

} // namespace lldb_private

#endif  // liblldb_Process_h_
