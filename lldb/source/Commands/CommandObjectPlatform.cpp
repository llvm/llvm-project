//===-- CommandObjectPlatform.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectPlatform.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/OptionGroupPlatform.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/Process.h"

using namespace lldb;
using namespace lldb_private;


//----------------------------------------------------------------------
// "platform select <platform-name>"
//----------------------------------------------------------------------
class CommandObjectPlatformSelect : public CommandObjectParsed
{
public:
    CommandObjectPlatformSelect (CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter, 
                             "platform select",
                             "Create a platform if needed and select it as the current platform.",
                             "platform select <platform-name>",
                             0),
        m_option_group (interpreter),
        m_platform_options (false) // Don't include the "--platform" option by passing false
    {
        m_option_group.Append (&m_platform_options, LLDB_OPT_SET_ALL, 1);
        m_option_group.Finalize();
    }

    virtual
    ~CommandObjectPlatformSelect ()
    {
    }

    virtual int
    HandleCompletion (Args &input,
                      int &cursor_index,
                      int &cursor_char_position,
                      int match_start_point,
                      int max_return_elements,
                      bool &word_complete,
                      StringList &matches)
    {
        std::string completion_str (input.GetArgumentAtIndex(cursor_index));
        completion_str.erase (cursor_char_position);
        
        CommandCompletions::PlatformPluginNames (m_interpreter, 
                                                 completion_str.c_str(),
                                                 match_start_point,
                                                 max_return_elements,
                                                 NULL,
                                                 word_complete,
                                                 matches);
        return matches.GetSize();
    }

    virtual Options *
    GetOptions ()
    {
        return &m_option_group;
    }

protected:
    virtual bool
    DoExecute (Args& args, CommandReturnObject &result)
    {
        if (args.GetArgumentCount() == 1)
        {
            const char *platform_name = args.GetArgumentAtIndex (0);
            if (platform_name && platform_name[0])
            {
                const bool select = true;
                m_platform_options.SetPlatformName (platform_name);
                Error error;
                ArchSpec platform_arch;
                PlatformSP platform_sp (m_platform_options.CreatePlatformWithOptions (m_interpreter, ArchSpec(), select, error, platform_arch));
                if (platform_sp)
                {
                    platform_sp->GetStatus (result.GetOutputStream());
                    result.SetStatus (eReturnStatusSuccessFinishResult);
                }
                else
                {
                    result.AppendError(error.AsCString());
                    result.SetStatus (eReturnStatusFailed);
                }
            }
            else
            {
                result.AppendError ("invalid platform name");
                result.SetStatus (eReturnStatusFailed);
            }
        }
        else
        {
            result.AppendError ("platform create takes a platform name as an argument\n");
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }

    OptionGroupOptions m_option_group;
    OptionGroupPlatform m_platform_options;
};

//----------------------------------------------------------------------
// "platform list"
//----------------------------------------------------------------------
class CommandObjectPlatformList : public CommandObjectParsed
{
public:
    CommandObjectPlatformList (CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter,
                             "platform list",
                             "List all platforms that are available.",
                             NULL,
                             0)
    {
    }

    virtual
    ~CommandObjectPlatformList ()
    {
    }

protected:
    virtual bool
    DoExecute (Args& args, CommandReturnObject &result)
    {
        Stream &ostrm = result.GetOutputStream();
        ostrm.Printf("Available platforms:\n");
        
        PlatformSP host_platform_sp (Platform::GetDefaultPlatform());
        ostrm.Printf ("%s: %s\n", 
                      host_platform_sp->GetShortPluginName(), 
                      host_platform_sp->GetDescription());

        uint32_t idx;
        for (idx = 0; 1; ++idx)
        {
            const char *plugin_name = PluginManager::GetPlatformPluginNameAtIndex (idx);
            if (plugin_name == NULL)
                break;
            const char *plugin_desc = PluginManager::GetPlatformPluginDescriptionAtIndex (idx);
            if (plugin_desc == NULL)
                break;
            ostrm.Printf("%s: %s\n", plugin_name, plugin_desc);
        }
        
        if (idx == 0)
        {
            result.AppendError ("no platforms are available\n");
            result.SetStatus (eReturnStatusFailed);
        }
        else
            result.SetStatus (eReturnStatusSuccessFinishResult);
        return result.Succeeded();
    }
};

//----------------------------------------------------------------------
// "platform status"
//----------------------------------------------------------------------
class CommandObjectPlatformStatus : public CommandObjectParsed
{
public:
    CommandObjectPlatformStatus (CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter,
                             "platform status",
                             "Display status for the currently selected platform.",
                             NULL,
                             0)
    {
    }

    virtual
    ~CommandObjectPlatformStatus ()
    {
    }

protected:
    virtual bool
    DoExecute (Args& args, CommandReturnObject &result)
    {
        Stream &ostrm = result.GetOutputStream();      
        
        PlatformSP platform_sp (m_interpreter.GetDebugger().GetPlatformList().GetSelectedPlatform());
        if (platform_sp)
        {
            platform_sp->GetStatus (ostrm);
            result.SetStatus (eReturnStatusSuccessFinishResult);            
        }
        else
        {
            result.AppendError ("no platform us currently selected\n");
            result.SetStatus (eReturnStatusFailed);            
        }
        return result.Succeeded();
    }
};

//----------------------------------------------------------------------
// "platform connect <connect-url>"
//----------------------------------------------------------------------
class CommandObjectPlatformConnect : public CommandObjectParsed
{
public:
    CommandObjectPlatformConnect (CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter, 
                             "platform connect",
                             "Connect a platform by name to be the currently selected platform.",
                             "platform connect <connect-url>",
                             0)
    {
    }

    virtual
    ~CommandObjectPlatformConnect ()
    {
    }

protected:
    virtual bool
    DoExecute (Args& args, CommandReturnObject &result)
    {
        Stream &ostrm = result.GetOutputStream();      
        
        PlatformSP platform_sp (m_interpreter.GetDebugger().GetPlatformList().GetSelectedPlatform());
        if (platform_sp)
        {
            Error error (platform_sp->ConnectRemote (args));
            if (error.Success())
            {
                platform_sp->GetStatus (ostrm);
                result.SetStatus (eReturnStatusSuccessFinishResult);            
            }
            else
            {
                result.AppendErrorWithFormat ("%s\n", error.AsCString());
                result.SetStatus (eReturnStatusFailed);            
            }
        }
        else
        {
            result.AppendError ("no platform us currently selected\n");
            result.SetStatus (eReturnStatusFailed);            
        }
        return result.Succeeded();
    }
};

//----------------------------------------------------------------------
// "platform disconnect"
//----------------------------------------------------------------------
class CommandObjectPlatformDisconnect : public CommandObjectParsed
{
public:
    CommandObjectPlatformDisconnect (CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter, 
                             "platform disconnect",
                             "Disconnect a platform by name to be the currently selected platform.",
                             "platform disconnect",
                             0)
    {
    }

    virtual
    ~CommandObjectPlatformDisconnect ()
    {
    }

protected:
    virtual bool
    DoExecute (Args& args, CommandReturnObject &result)
    {
        PlatformSP platform_sp (m_interpreter.GetDebugger().GetPlatformList().GetSelectedPlatform());
        if (platform_sp)
        {
            if (args.GetArgumentCount() == 0)
            {
                Error error;
                
                if (platform_sp->IsConnected())
                {
                    // Cache the instance name if there is one since we are 
                    // about to disconnect and the name might go with it.
                    const char *hostname_cstr = platform_sp->GetHostname();
                    std::string hostname;
                    if (hostname_cstr)
                        hostname.assign (hostname_cstr);

                    error = platform_sp->DisconnectRemote ();
                    if (error.Success())
                    {
                        Stream &ostrm = result.GetOutputStream();      
                        if (hostname.empty())
                            ostrm.Printf ("Disconnected from \"%s\"\n", platform_sp->GetShortPluginName());
                        else
                            ostrm.Printf ("Disconnected from \"%s\"\n", hostname.c_str());
                        result.SetStatus (eReturnStatusSuccessFinishResult);            
                    }
                    else
                    {
                        result.AppendErrorWithFormat ("%s", error.AsCString());
                        result.SetStatus (eReturnStatusFailed);            
                    }
                }
                else
                {
                    // Not connected...
                    result.AppendErrorWithFormat ("not connected to '%s'", platform_sp->GetShortPluginName());
                    result.SetStatus (eReturnStatusFailed);            
                }
            }
            else
            {
                // Bad args
                result.AppendError ("\"platform disconnect\" doesn't take any arguments");
                result.SetStatus (eReturnStatusFailed);            
            }
        }
        else
        {
            result.AppendError ("no platform is currently selected");
            result.SetStatus (eReturnStatusFailed);            
        }
        return result.Succeeded();
    }
};
//----------------------------------------------------------------------
// "platform process launch"
//----------------------------------------------------------------------
class CommandObjectPlatformProcessLaunch : public CommandObjectParsed
{
public:
    CommandObjectPlatformProcessLaunch (CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter, 
                             "platform process launch",
                             "Launch a new process on a remote platform.",
                             "platform process launch program",
                             0),
        m_options (interpreter)
    {
    }
    
    virtual
    ~CommandObjectPlatformProcessLaunch ()
    {
    }
    
    virtual Options *
    GetOptions ()
    {
        return &m_options;
    }
    
protected:
    virtual bool
    DoExecute (Args& args, CommandReturnObject &result)
    {
        PlatformSP platform_sp (m_interpreter.GetDebugger().GetPlatformList().GetSelectedPlatform());
        
        if (platform_sp)
        {
            Error error;
            const uint32_t argc = args.GetArgumentCount();
            Target *target = m_interpreter.GetExecutionContext().GetTargetPtr();
            if (target == NULL)
            {
                result.AppendError ("invalid target, create a debug target using the 'target create' command");
                result.SetStatus (eReturnStatusFailed);
                return false;
            }

            Module *exe_module = target->GetExecutableModulePointer();
            if (exe_module)
            {
                m_options.launch_info.GetExecutableFile () = exe_module->GetFileSpec();
                char exe_path[PATH_MAX];
                if (m_options.launch_info.GetExecutableFile ().GetPath (exe_path, sizeof(exe_path)))
                    m_options.launch_info.GetArguments().AppendArgument (exe_path);
                m_options.launch_info.GetArchitecture() = exe_module->GetArchitecture();
            }

            if (argc > 0)
            {
                if (m_options.launch_info.GetExecutableFile ())
                {
                    // We already have an executable file, so we will use this
                    // and all arguments to this function are extra arguments
                    m_options.launch_info.GetArguments().AppendArguments (args);
                }
                else
                {
                    // We don't have any file yet, so the first argument is our
                    // executable, and the rest are program arguments
                    const bool first_arg_is_executable = true;
                    m_options.launch_info.SetArguments (args, first_arg_is_executable);
                }
            }
            
            if (m_options.launch_info.GetExecutableFile ())
            {
                Debugger &debugger = m_interpreter.GetDebugger();

                if (argc == 0)
                    target->GetRunArguments(m_options.launch_info.GetArguments());

                ProcessSP process_sp (platform_sp->DebugProcess (m_options.launch_info, 
                                                                 debugger,
                                                                 target,
                                                                 debugger.GetListener(),
                                                                 error));
                if (process_sp && process_sp->IsAlive())
                {
                    result.SetStatus (eReturnStatusSuccessFinishNoResult);
                    return true;
                }
                
                if (error.Success())
                    result.AppendError ("process launch failed");
                else
                    result.AppendError (error.AsCString());
                result.SetStatus (eReturnStatusFailed);
            }
            else
            {
                result.AppendError ("'platform process launch' uses the current target file and arguments, or the executable and its arguments can be specified in this command");
                result.SetStatus (eReturnStatusFailed);
                return false;
            }
        }
        else
        {
            result.AppendError ("no platform is selected\n");
        }
        return result.Succeeded();
    }
    
protected:
    ProcessLaunchCommandOptions m_options;
};



//----------------------------------------------------------------------
// "platform process list"
//----------------------------------------------------------------------
class CommandObjectPlatformProcessList : public CommandObjectParsed
{
public:
    CommandObjectPlatformProcessList (CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter, 
                             "platform process list",
                             "List processes on a remote platform by name, pid, or many other matching attributes.",
                             "platform process list",
                             0),
        m_options (interpreter)
    {
    }
    
    virtual
    ~CommandObjectPlatformProcessList ()
    {
    }
    
    virtual Options *
    GetOptions ()
    {
        return &m_options;
    }
    
protected:
    virtual bool
    DoExecute (Args& args, CommandReturnObject &result)
    {
        PlatformSP platform_sp (m_interpreter.GetDebugger().GetPlatformList().GetSelectedPlatform());
        
        if (platform_sp)
        {
            Error error;
            if (args.GetArgumentCount() == 0)
            {
                
                if (platform_sp)
                {
                    Stream &ostrm = result.GetOutputStream();      

                    lldb::pid_t pid = m_options.match_info.GetProcessInfo().GetProcessID();
                    if (pid != LLDB_INVALID_PROCESS_ID)
                    {
                        ProcessInstanceInfo proc_info;
                        if (platform_sp->GetProcessInfo (pid, proc_info))
                        {
                            ProcessInstanceInfo::DumpTableHeader (ostrm, platform_sp.get(), m_options.show_args, m_options.verbose);
                            proc_info.DumpAsTableRow(ostrm, platform_sp.get(), m_options.show_args, m_options.verbose);
                            result.SetStatus (eReturnStatusSuccessFinishResult);
                        }
                        else
                        {
                            result.AppendErrorWithFormat ("no process found with pid = %llu\n", pid);
                            result.SetStatus (eReturnStatusFailed);
                        }
                    }
                    else
                    {
                        ProcessInstanceInfoList proc_infos;
                        const uint32_t matches = platform_sp->FindProcesses (m_options.match_info, proc_infos);
                        const char *match_desc = NULL;
                        const char *match_name = m_options.match_info.GetProcessInfo().GetName();
                        if (match_name && match_name[0])
                        {
                            switch (m_options.match_info.GetNameMatchType())
                            {
                                case eNameMatchIgnore: break;
                                case eNameMatchEquals: match_desc = "matched"; break;
                                case eNameMatchContains: match_desc = "contained"; break;
                                case eNameMatchStartsWith: match_desc = "started with"; break;
                                case eNameMatchEndsWith: match_desc = "ended with"; break;
                                case eNameMatchRegularExpression: match_desc = "matched the regular expression"; break;
                            }
                        }

                        if (matches == 0)
                        {
                            if (match_desc)
                                result.AppendErrorWithFormat ("no processes were found that %s \"%s\" on the \"%s\" platform\n", 
                                                              match_desc,
                                                              match_name,
                                                              platform_sp->GetShortPluginName());
                            else
                                result.AppendErrorWithFormat ("no processes were found on the \"%s\" platform\n", platform_sp->GetShortPluginName());
                            result.SetStatus (eReturnStatusFailed);
                        }
                        else
                        {
                            result.AppendMessageWithFormat ("%u matching process%s found on \"%s\"", 
                                                            matches,
                                                            matches > 1 ? "es were" : " was",
                                                            platform_sp->GetName());
                            if (match_desc)
                                result.AppendMessageWithFormat (" whose name %s \"%s\"", 
                                                                match_desc,
                                                                match_name);
                            result.AppendMessageWithFormat ("\n");
                            ProcessInstanceInfo::DumpTableHeader (ostrm, platform_sp.get(), m_options.show_args, m_options.verbose);
                            for (uint32_t i=0; i<matches; ++i)
                            {
                                proc_infos.GetProcessInfoAtIndex(i).DumpAsTableRow(ostrm, platform_sp.get(), m_options.show_args, m_options.verbose);
                            }
                        }
                    }
                }
            }
            else
            {
                result.AppendError ("invalid args: process list takes only options\n");
                result.SetStatus (eReturnStatusFailed);
            }
        }
        else
        {
            result.AppendError ("no platform is selected\n");
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }
    
    class CommandOptions : public Options
    {
    public:
        
        CommandOptions (CommandInterpreter &interpreter) :
            Options (interpreter),
            match_info ()
        {
        }
        
        virtual
        ~CommandOptions ()
        {
        }
        
        virtual Error
        SetOptionValue (uint32_t option_idx, const char *option_arg)
        {
            Error error;
            char short_option = (char) m_getopt_table[option_idx].val;
            bool success = false;

            switch (short_option)
            {
                case 'p':
                    match_info.GetProcessInfo().SetProcessID (Args::StringToUInt32 (option_arg, LLDB_INVALID_PROCESS_ID, 0, &success));
                    if (!success)
                        error.SetErrorStringWithFormat("invalid process ID string: '%s'", option_arg);
                    break;
                
                case 'P':
                    match_info.GetProcessInfo().SetParentProcessID (Args::StringToUInt32 (option_arg, LLDB_INVALID_PROCESS_ID, 0, &success));
                    if (!success)
                        error.SetErrorStringWithFormat("invalid parent process ID string: '%s'", option_arg);
                    break;

                case 'u':
                    match_info.GetProcessInfo().SetUserID (Args::StringToUInt32 (option_arg, UINT32_MAX, 0, &success));
                    if (!success)
                        error.SetErrorStringWithFormat("invalid user ID string: '%s'", option_arg);
                    break;

                case 'U':
                    match_info.GetProcessInfo().SetEffectiveUserID (Args::StringToUInt32 (option_arg, UINT32_MAX, 0, &success));
                    if (!success)
                        error.SetErrorStringWithFormat("invalid effective user ID string: '%s'", option_arg);
                    break;

                case 'g':
                    match_info.GetProcessInfo().SetGroupID (Args::StringToUInt32 (option_arg, UINT32_MAX, 0, &success));
                    if (!success)
                        error.SetErrorStringWithFormat("invalid group ID string: '%s'", option_arg);
                    break;

                case 'G':
                    match_info.GetProcessInfo().SetEffectiveGroupID (Args::StringToUInt32 (option_arg, UINT32_MAX, 0, &success));
                    if (!success)
                        error.SetErrorStringWithFormat("invalid effective group ID string: '%s'", option_arg);
                    break;

                case 'a':
                    match_info.GetProcessInfo().GetArchitecture().SetTriple (option_arg, m_interpreter.GetDebugger().GetPlatformList().GetSelectedPlatform().get());
                    break;

                case 'n':
                    match_info.GetProcessInfo().GetExecutableFile().SetFile (option_arg, false);
                    match_info.SetNameMatchType (eNameMatchEquals);
                    break;

                case 'e':
                    match_info.GetProcessInfo().GetExecutableFile().SetFile (option_arg, false);
                    match_info.SetNameMatchType (eNameMatchEndsWith);
                    break;

                case 's':
                    match_info.GetProcessInfo().GetExecutableFile().SetFile (option_arg, false);
                    match_info.SetNameMatchType (eNameMatchStartsWith);
                    break;
                    
                case 'c':
                    match_info.GetProcessInfo().GetExecutableFile().SetFile (option_arg, false);
                    match_info.SetNameMatchType (eNameMatchContains);
                    break;
                    
                case 'r':
                    match_info.GetProcessInfo().GetExecutableFile().SetFile (option_arg, false);
                    match_info.SetNameMatchType (eNameMatchRegularExpression);
                    break;

                case 'A':
                    show_args = true;
                    break;

                case 'v':
                    verbose = true;
                    break;

                default:
                    error.SetErrorStringWithFormat ("unrecognized option '%c'", short_option);
                    break;
            }
            
            return error;
        }
        
        void
        OptionParsingStarting ()
        {
            match_info.Clear();
            show_args = false;
            verbose = false;
        }
        
        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }
        
        // Options table: Required for subclasses of Options.
        
        static OptionDefinition g_option_table[];
        
        // Instance variables to hold the values for command options.
        
        ProcessInstanceInfoMatch match_info;
        bool show_args;
        bool verbose;
    };
    CommandOptions m_options;
};

OptionDefinition
CommandObjectPlatformProcessList::CommandOptions::g_option_table[] =
{
{   LLDB_OPT_SET_1, false, "pid"              , 'p', required_argument, NULL, 0, eArgTypePid          , "List the process info for a specific process ID." },
{   LLDB_OPT_SET_2, true , "name"             , 'n', required_argument, NULL, 0, eArgTypeProcessName  , "Find processes with executable basenames that match a string." },
{   LLDB_OPT_SET_3, true , "ends-with"        , 'e', required_argument, NULL, 0, eArgTypeNone         , "Find processes with executable basenames that end with a string." },
{   LLDB_OPT_SET_4, true , "starts-with"      , 's', required_argument, NULL, 0, eArgTypeNone         , "Find processes with executable basenames that start with a string." },
{   LLDB_OPT_SET_5, true , "contains"         , 'c', required_argument, NULL, 0, eArgTypeNone         , "Find processes with executable basenames that contain a string." },
{   LLDB_OPT_SET_6, true , "regex"            , 'r', required_argument, NULL, 0, eArgTypeNone         , "Find processes with executable basenames that match a regular expression." },
{  ~LLDB_OPT_SET_1, false, "parent"           , 'P', required_argument, NULL, 0, eArgTypePid          , "Find processes that have a matching parent process ID." },
{  ~LLDB_OPT_SET_1, false, "uid"              , 'u', required_argument, NULL, 0, eArgTypeNone         , "Find processes that have a matching user ID." },
{  ~LLDB_OPT_SET_1, false, "euid"             , 'U', required_argument, NULL, 0, eArgTypeNone         , "Find processes that have a matching effective user ID." },
{  ~LLDB_OPT_SET_1, false, "gid"              , 'g', required_argument, NULL, 0, eArgTypeNone         , "Find processes that have a matching group ID." },
{  ~LLDB_OPT_SET_1, false, "egid"             , 'G', required_argument, NULL, 0, eArgTypeNone         , "Find processes that have a matching effective group ID." },
{  ~LLDB_OPT_SET_1, false, "arch"             , 'a', required_argument, NULL, 0, eArgTypeArchitecture , "Find processes that have a matching architecture." },
{ LLDB_OPT_SET_ALL, false, "show-args"        , 'A', no_argument      , NULL, 0, eArgTypeNone         , "Show process arguments instead of the process executable basename." },
{ LLDB_OPT_SET_ALL, false, "verbose"          , 'v', no_argument      , NULL, 0, eArgTypeNone         , "Enable verbose output." },
{  0              , false, NULL               ,  0 , 0                , NULL, 0, eArgTypeNone         , NULL }
};

//----------------------------------------------------------------------
// "platform process info"
//----------------------------------------------------------------------
class CommandObjectPlatformProcessInfo : public CommandObjectParsed
{
public:
    CommandObjectPlatformProcessInfo (CommandInterpreter &interpreter) :
    CommandObjectParsed (interpreter, 
                         "platform process info",
                         "Get detailed information for one or more process by process ID.",
                         "platform process info <pid> [<pid> <pid> ...]",
                         0)
    {
        CommandArgumentEntry arg;
        CommandArgumentData pid_args;
        
        // Define the first (and only) variant of this arg.
        pid_args.arg_type = eArgTypePid;
        pid_args.arg_repetition = eArgRepeatStar;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg.push_back (pid_args);
        
        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg);
    }
    
    virtual
    ~CommandObjectPlatformProcessInfo ()
    {
    }
    
protected:
    virtual bool
    DoExecute (Args& args, CommandReturnObject &result)
    {
        PlatformSP platform_sp (m_interpreter.GetDebugger().GetPlatformList().GetSelectedPlatform());
        if (platform_sp)
        {
            const size_t argc = args.GetArgumentCount();
            if (argc > 0)
            {
                Error error;
                
                if (platform_sp->IsConnected())
                {
                    Stream &ostrm = result.GetOutputStream();      
                    bool success;
                    for (size_t i=0; i<argc; ++ i)
                    {
                        const char *arg = args.GetArgumentAtIndex(i);
                        lldb::pid_t pid = Args::StringToUInt32 (arg, LLDB_INVALID_PROCESS_ID, 0, &success);
                        if (success)
                        {
                            ProcessInstanceInfo proc_info;
                            if (platform_sp->GetProcessInfo (pid, proc_info))
                            {
                                ostrm.Printf ("Process information for process %llu:\n", pid);
                                proc_info.Dump (ostrm, platform_sp.get());
                            }
                            else
                            {
                                ostrm.Printf ("error: no process information is available for process %llu\n", pid);
                            }
                            ostrm.EOL();
                        }
                        else
                        {
                            result.AppendErrorWithFormat ("invalid process ID argument '%s'", arg);
                            result.SetStatus (eReturnStatusFailed);            
                            break;
                        }
                    }
                }
                else
                {
                    // Not connected...
                    result.AppendErrorWithFormat ("not connected to '%s'", platform_sp->GetShortPluginName());
                    result.SetStatus (eReturnStatusFailed);            
                }
            }
            else
            {
                // No args
                result.AppendError ("one or more process id(s) must be specified");
                result.SetStatus (eReturnStatusFailed);            
            }
        }
        else
        {
            result.AppendError ("no platform is currently selected");
            result.SetStatus (eReturnStatusFailed);            
        }
        return result.Succeeded();
    }
};




class CommandObjectPlatformProcess : public CommandObjectMultiword
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
     CommandObjectPlatformProcess (CommandInterpreter &interpreter) :
        CommandObjectMultiword (interpreter,
                                "platform process",
                                "A set of commands to query, launch and attach to platform processes",
                                "platform process [attach|launch|list] ...")
    {
//        LoadSubCommand ("attach", CommandObjectSP (new CommandObjectPlatformProcessAttach (interpreter)));
        LoadSubCommand ("launch", CommandObjectSP (new CommandObjectPlatformProcessLaunch (interpreter)));
        LoadSubCommand ("info"  , CommandObjectSP (new CommandObjectPlatformProcessInfo (interpreter)));
        LoadSubCommand ("list"  , CommandObjectSP (new CommandObjectPlatformProcessList (interpreter)));

    }
    
    virtual
    ~CommandObjectPlatformProcess ()
    {
    }
    
private:
    //------------------------------------------------------------------
    // For CommandObjectPlatform only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (CommandObjectPlatformProcess);
};


class CommandObjectPlatformShell : public CommandObjectRaw
{
public:
    CommandObjectPlatformShell (CommandInterpreter &interpreter) :
        CommandObjectRaw (interpreter, 
                         "platform shell",
                         "Run a shell command on a the selected platform.",
                         "platform shell <shell-command>",
                         0)
    {
    }
    
    virtual
    ~CommandObjectPlatformShell ()
    {
    }
    
protected:
    virtual bool
    DoExecute (const char *raw_command_line, CommandReturnObject &result)
    {
        // TODO: Implement "Platform::RunShellCommand()" and switch over to using
        // the current platform when it is in the interface. 
        const char *working_dir = NULL;
        std::string output;
        int status = -1;
        int signo = -1;
        Error error (Host::RunShellCommand (raw_command_line, working_dir, &status, &signo, &output, 10));
        if (!output.empty())
            result.GetOutputStream().PutCString(output.c_str());
        if (status > 0)
        {
            if (signo > 0)
            {
                const char *signo_cstr = Host::GetSignalAsCString(signo);
                if (signo_cstr)
                    result.GetOutputStream().Printf("error: command returned with status %i and signal %s\n", status, signo_cstr);
                else
                    result.GetOutputStream().Printf("error: command returned with status %i and signal %i\n", status, signo);
            }
            else
                result.GetOutputStream().Printf("error: command returned with status %i\n", status);
        }

        if (error.Fail())
        {
            result.AppendError(error.AsCString());
            result.SetStatus (eReturnStatusFailed);
        }
        else
        {
            result.SetStatus (eReturnStatusSuccessFinishResult);
        }
        return true;
    }
};

//----------------------------------------------------------------------
// CommandObjectPlatform constructor
//----------------------------------------------------------------------
CommandObjectPlatform::CommandObjectPlatform(CommandInterpreter &interpreter) :
    CommandObjectMultiword (interpreter,
                            "platform",
                            "A set of commands to manage and create platforms.",
                            "platform [connect|disconnect|info|list|status|select] ...")
{
    LoadSubCommand ("select", CommandObjectSP (new CommandObjectPlatformSelect  (interpreter)));
    LoadSubCommand ("list"  , CommandObjectSP (new CommandObjectPlatformList    (interpreter)));
    LoadSubCommand ("status", CommandObjectSP (new CommandObjectPlatformStatus  (interpreter)));
    LoadSubCommand ("connect", CommandObjectSP (new CommandObjectPlatformConnect  (interpreter)));
    LoadSubCommand ("disconnect", CommandObjectSP (new CommandObjectPlatformDisconnect  (interpreter)));
    LoadSubCommand ("process", CommandObjectSP (new CommandObjectPlatformProcess  (interpreter)));
    LoadSubCommand ("shell", CommandObjectSP (new CommandObjectPlatformShell  (interpreter)));
}


//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
CommandObjectPlatform::~CommandObjectPlatform()
{
}
