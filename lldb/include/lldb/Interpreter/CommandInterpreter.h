//===-- CommandInterpreter.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandInterpreter_h_
#define liblldb_CommandInterpreter_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Core/Broadcaster.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Log.h"
#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/Core/Event.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Core/StringList.h"

namespace lldb_private {

class CommandInterpreter :
    public Broadcaster,
    public Properties
{
public:
    typedef std::map<std::string, OptionArgVectorSP> OptionArgMap;

    enum
    {
        eBroadcastBitThreadShouldExit       = (1 << 0),
        eBroadcastBitResetPrompt            = (1 << 1),
        eBroadcastBitQuitCommandReceived    = (1 << 2),   // User entered quit
        eBroadcastBitAsynchronousOutputData = (1 << 3),
        eBroadcastBitAsynchronousErrorData  = (1 << 4)
    };
    
    enum ChildrenTruncatedWarningStatus // tristate boolean to manage children truncation warning
    {
        eNoTruncation = 0, // never truncated
        eUnwarnedTruncation = 1, // truncated but did not notify
        eWarnedTruncation = 2 // truncated and notified
    };
    
    enum CommandTypes
    {
        eCommandTypesBuiltin = 0x0001,  // native commands such as "frame"
        eCommandTypesUserDef = 0x0002,  // scripted commands
        eCommandTypesAliases = 0x0004,  // aliases such as "po"
        eCommandTypesAllThem = 0xFFFF   // all commands
    };

    // These two functions fill out the Broadcaster interface:
    
    static ConstString &GetStaticBroadcasterClass ();

    virtual ConstString &GetBroadcasterClass() const
    {
        return GetStaticBroadcasterClass();
    }

    void
    SourceInitFile (bool in_cwd, 
                    CommandReturnObject &result);

    CommandInterpreter (Debugger &debugger,
                        lldb::ScriptLanguage script_language,
                        bool synchronous_execution);

    virtual
    ~CommandInterpreter ();

    bool
    AddCommand (const char *name, 
                const lldb::CommandObjectSP &cmd_sp,
                bool can_replace);
    
    bool
    AddUserCommand (std::string name, 
                    const lldb::CommandObjectSP &cmd_sp,
                    bool can_replace);
    
    lldb::CommandObjectSP
    GetCommandSPExact (const char *cmd, 
                       bool include_aliases);

    CommandObject *
    GetCommandObjectExact (const char *cmd_cstr, 
                           bool include_aliases);

    CommandObject *
    GetCommandObject (const char *cmd, 
                      StringList *matches = NULL);

    bool
    CommandExists (const char *cmd);

    bool
    AliasExists (const char *cmd);

    bool
    UserCommandExists (const char *cmd);

    void
    AddAlias (const char *alias_name, 
              lldb::CommandObjectSP& command_obj_sp);

    bool
    RemoveAlias (const char *alias_name);

    bool
    RemoveUser (const char *alias_name);
    
    void
    RemoveAllUser ()
    {
        m_user_dict.clear();
    }

    OptionArgVectorSP
    GetAliasOptions (const char *alias_name);


    bool
    ProcessAliasOptionsArgs (lldb::CommandObjectSP &cmd_obj_sp, 
                             const char *options_args,
                             OptionArgVectorSP &option_arg_vector_sp);

    void
    RemoveAliasOptions (const char *alias_name);

    void
    AddOrReplaceAliasOptions (const char *alias_name, 
                              OptionArgVectorSP &option_arg_vector_sp);

    CommandObject *
    BuildAliasResult (const char *alias_name, 
                      std::string &raw_input_string, 
                      std::string &alias_result, 
                      CommandReturnObject &result);

    bool
    HandleCommand (const char *command_line, 
                   LazyBool add_to_history,
                   CommandReturnObject &result, 
                   ExecutionContext *override_context = NULL,
                   bool repeat_on_empty_command = true,
                   bool no_context_switching = false);
    
    //------------------------------------------------------------------
    /// Execute a list of commands in sequence.
    ///
    /// @param[in] commands
    ///    The list of commands to execute.
    /// @param[in/out] context 
    ///    The execution context in which to run the commands.  Can be NULL in which case the default
    ///    context will be used.
    /// @param[in] stop_on_continue 
    ///    If \b true execution will end on the first command that causes the process in the
    ///    execution context to continue.  If \false, we won't check the execution status.
    /// @param[in] stop_on_error 
    ///    If \b true execution will end on the first command that causes an error.
    /// @param[in] echo_commands
    ///    If \b true echo the command before executing it.  If \false, execute silently.
    /// @param[in] print_results
    ///    If \b true print the results of the command after executing it.  If \false, execute silently.
    /// @param[out] result 
    ///    This is marked as succeeding with no output if all commands execute safely,
    ///    and failed with some explanation if we aborted executing the commands at some point.
    //------------------------------------------------------------------
    void
    HandleCommands (const StringList &commands, 
                    ExecutionContext *context, 
                    bool stop_on_continue, 
                    bool stop_on_error, 
                    bool echo_commands,
                    bool print_results,
                    LazyBool add_to_history,
                    CommandReturnObject &result);

    //------------------------------------------------------------------
    /// Execute a list of commands from a file.
    ///
    /// @param[in] file
    ///    The file from which to read in commands.
    /// @param[in/out] context 
    ///    The execution context in which to run the commands.  Can be NULL in which case the default
    ///    context will be used.
    /// @param[in] stop_on_continue 
    ///    If \b true execution will end on the first command that causes the process in the
    ///    execution context to continue.  If \false, we won't check the execution status.
    /// @param[in] stop_on_error 
    ///    If \b true execution will end on the first command that causes an error.
    /// @param[in] echo_commands
    ///    If \b true echo the command before executing it.  If \false, execute silently.
    /// @param[in] print_results
    ///    If \b true print the results of the command after executing it.  If \false, execute silently.
    /// @param[out] result 
    ///    This is marked as succeeding with no output if all commands execute safely,
    ///    and failed with some explanation if we aborted executing the commands at some point.
    //------------------------------------------------------------------
    void
    HandleCommandsFromFile (FileSpec &file, 
                            ExecutionContext *context, 
                            bool stop_on_continue, 
                            bool stop_on_error, 
                            bool echo_commands,
                            bool print_results,
                            LazyBool add_to_history,
                            CommandReturnObject &result);

    CommandObject *
    GetCommandObjectForCommand (std::string &command_line);

    // This handles command line completion.  You are given a pointer to the command string buffer, to the current cursor,
    // and to the end of the string (in case it is not NULL terminated).
    // You also passed in an StringList object to fill with the returns.
    // The first element of the array will be filled with the string that you would need to insert at
    // the cursor point to complete the cursor point to the longest common matching prefix.
    // If you want to limit the number of elements returned, set max_return_elements to the number of elements
    // you want returned.  Otherwise set max_return_elements to -1.
    // If you want to start some way into the match list, then set match_start_point to the desired start
    // point.
    // Returns:
    // -1 if the completion character should be inserted
    // -2 if the entire command line should be deleted and replaced with matches.GetStringAtIndex(0)
    // INT_MAX if the number of matches is > max_return_elements, but it is expensive to compute.
    // Otherwise, returns the number of matches.
    //
    // FIXME: Only max_return_elements == -1 is supported at present.

    int
    HandleCompletion (const char *current_line,
                      const char *cursor,
                      const char *last_char,
                      int match_start_point,
                      int max_return_elements,
                      StringList &matches);

    // This version just returns matches, and doesn't compute the substring.  It is here so the
    // Help command can call it for the first argument.
    // word_complete tells whether a the completions are considered a "complete" response (so the
    // completer should complete the quote & put a space after the word.

    int
    HandleCompletionMatches (Args &input,
                             int &cursor_index,
                             int &cursor_char_position,
                             int match_start_point,
                             int max_return_elements,
                             bool &word_complete,
                             StringList &matches);


    int
    GetCommandNamesMatchingPartialString (const char *cmd_cstr, 
                                          bool include_aliases, 
                                          StringList &matches);

    void
    GetHelp (CommandReturnObject &result,
             uint32_t types = eCommandTypesAllThem);

    void
    GetAliasHelp (const char *alias_name, 
                  const char *command_name, 
                  StreamString &help_string);

    void
    OutputFormattedHelpText (Stream &stream,
                             const char *command_word,
                             const char *separator,
                             const char *help_text,
                             uint32_t max_word_len);
    
    // this mimics OutputFormattedHelpText but it does perform a much simpler
    // formatting, basically ensuring line alignment. This is only good if you have
    // some complicated layout for your help text and want as little help as reasonable
    // in properly displaying it. Most of the times, you simply want to type some text
    // and have it printed in a reasonable way on screen. If so, use OutputFormattedHelpText 
    void
    OutputHelpText (Stream &stream,
                    const char *command_word,
                    const char *separator,
                    const char *help_text,
                    uint32_t max_word_len);

    Debugger &
    GetDebugger ()
    {
        return m_debugger;
    }
    
    ExecutionContext
    GetExecutionContext()
    {
        return m_exe_ctx_ref.Lock();
    }
    
    void
    UpdateExecutionContext (ExecutionContext *override_context);

    lldb::PlatformSP
    GetPlatform (bool prefer_target_platform);

    const char *
    ProcessEmbeddedScriptCommands (const char *arg);

    const char *
    GetPrompt ();

    void
    SetPrompt (const char *);
    
    bool Confirm (const char *message, bool default_answer);
    
    static size_t
    GetConfirmationInputReaderCallback (void *baton,
                                        InputReader &reader,
                                        lldb::InputReaderAction action,
                                        const char *bytes,
                                        size_t bytes_len);
    
    void
    LoadCommandDictionary ();

    void
    Initialize ();

    void
    CrossRegisterCommand (const char *dest_cmd, 
                          const char *object_type);

    void
    SetScriptLanguage (lldb::ScriptLanguage lang);


    bool
    HasCommands ();

    bool
    HasAliases ();

    bool
    HasUserCommands ();

    bool
    HasAliasOptions ();

    void
    BuildAliasCommandArgs (CommandObject *alias_cmd_obj, 
                           const char *alias_name, 
                           Args &cmd_args, 
                           std::string &raw_input_string, 
                           CommandReturnObject &result);

    int
    GetOptionArgumentPosition (const char *in_string);

    ScriptInterpreter *
    GetScriptInterpreter (bool can_create = true);

    void
    SkipLLDBInitFiles (bool skip_lldbinit_files)
    {
        m_skip_lldbinit_files = skip_lldbinit_files;
    }

    void
    SkipAppInitFiles (bool skip_app_init_files)
    {
        m_skip_app_init_files = m_skip_lldbinit_files;
    }

    bool
    GetSynchronous ();
    
    void
    DumpHistory (Stream &stream, uint32_t count) const;

    void
    DumpHistory (Stream &stream, uint32_t start, uint32_t end) const;
    
    const char *
    FindHistoryString (const char *input_str) const;


#ifndef SWIG
    void
    AddLogChannel (const char *name, 
                   const Log::Callbacks &log_callbacks);

    bool
    GetLogChannelCallbacks (const char *channel, 
                            Log::Callbacks &log_callbacks);

    bool
    RemoveLogChannel (const char *name);
#endif

    size_t
    FindLongestCommandWord (CommandObject::CommandMap &dict);

    void
    FindCommandsForApropos (const char *word, 
                            StringList &commands_found, 
                            StringList &commands_help);
                           
    bool
    GetBatchCommandMode () { return m_batch_command_mode; }
    
    void
    SetBatchCommandMode (bool value) { m_batch_command_mode = value; }
    
    void
    ChildrenTruncated()
    {
        if (m_truncation_warning == eNoTruncation)
            m_truncation_warning = eUnwarnedTruncation;
    }
    
    bool
    TruncationWarningNecessary()
    {
        return (m_truncation_warning == eUnwarnedTruncation);
    }
    
    void
    TruncationWarningGiven()
    {
        m_truncation_warning = eWarnedTruncation;
    }
    
    const char *
    TruncationWarningText()
    {
        return "*** Some of your variables have more members than the debugger will show by default. To show all of them, you can either use the --show-all-children option to %s or raise the limit by changing the target.max-children-count setting.\n";
    }
    
    //------------------------------------------------------------------
    // Properties
    //------------------------------------------------------------------
    bool
    GetExpandRegexAliases () const;
    
protected:
    friend class Debugger;

    void
    SetSynchronous (bool value);

    lldb::CommandObjectSP
    GetCommandSP (const char *cmd, bool include_aliases = true, bool exact = true, StringList *matches = NULL);

private:
    
    Error
    PreprocessCommand (std::string &command);

    Debugger &m_debugger;                       // The debugger session that this interpreter is associated with
    ExecutionContextRef m_exe_ctx_ref;          // The current execution context to use when handling commands
    bool m_synchronous_execution;
    bool m_skip_lldbinit_files;
    bool m_skip_app_init_files;
    CommandObject::CommandMap m_command_dict;   // Stores basic built-in commands (they cannot be deleted, removed or overwritten).
    CommandObject::CommandMap m_alias_dict;     // Stores user aliases/abbreviations for commands
    CommandObject::CommandMap m_user_dict;      // Stores user-defined commands
    OptionArgMap m_alias_options;               // Stores any options (with or without arguments) that go with any alias.
    std::vector<std::string> m_command_history;
    std::string m_repeat_command;               // Stores the command that will be executed for an empty command string.
    std::auto_ptr<ScriptInterpreter> m_script_interpreter_ap;
    char m_comment_char;
    char m_repeat_char;
    bool m_batch_command_mode;
    ChildrenTruncatedWarningStatus m_truncation_warning;    // Whether we truncated children and whether the user has been told
    uint32_t m_command_source_depth;
    
};


} // namespace lldb_private

#endif  // liblldb_CommandInterpreter_h_
