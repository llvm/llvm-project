//===-- CommandObject.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObject_h_
#define liblldb_CommandObject_h_

#include <map>
#include <set>
#include <string>
#include <vector>

#include "lldb/lldb-private.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Interpreter/CommandCompletions.h"
#include "lldb/Core/StringList.h"
#include "lldb/Core/Flags.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Target/ExecutionContext.h"

namespace lldb_private {

class CommandObject
{
public:

    typedef const char *(ArgumentHelpCallbackFunction) ();
    
    struct ArgumentHelpCallback
    {
        ArgumentHelpCallbackFunction  *help_callback;
        bool                           self_formatting;
        
        const char*
        operator () () const
        {
            return (*help_callback)();
        }
        
        operator bool() const
        {
            return (help_callback != NULL);
        }

    };
    
    struct ArgumentTableEntry  // Entries in the main argument information table
    {
        lldb::CommandArgumentType  arg_type;
        const char *arg_name;
        CommandCompletions::CommonCompletionTypes completion_type;
        ArgumentHelpCallback  help_function;
        const char *help_text;
    };
    
    struct CommandArgumentData  // Used to build individual command argument lists
    {
        lldb::CommandArgumentType arg_type;
        ArgumentRepetitionType arg_repetition;
        uint32_t arg_opt_set_association; // This arg might be associated only with some particular option set(s).
        CommandArgumentData():
            arg_type(lldb::eArgTypeNone),
            arg_repetition(eArgRepeatPlain),
            arg_opt_set_association(LLDB_OPT_SET_ALL) // By default, the arg associates to all option sets.
        {}
    };
    
    typedef std::vector<CommandArgumentData> CommandArgumentEntry; // Used to build individual command argument lists

    static ArgumentTableEntry g_arguments_data[lldb::eArgTypeLastArg];   // Main argument information table

    typedef std::map<std::string, lldb::CommandObjectSP> CommandMap;

    CommandObject (CommandInterpreter &interpreter,
                   const char *name,
                   const char *help = NULL,
                   const char *syntax = NULL,
                   uint32_t flags = 0);

    virtual
    ~CommandObject ();

    
    static const char * 
    GetArgumentTypeAsCString (const lldb::CommandArgumentType arg_type);
    
    static const char * 
    GetArgumentDescriptionAsCString (const lldb::CommandArgumentType arg_type);

    CommandInterpreter &
    GetCommandInterpreter ()
    {
        return m_interpreter;
    }

    const char *
    GetHelp ();

    virtual const char *
    GetHelpLong ();

    const char *
    GetSyntax ();

    const char *
    Translate ();

    const char *
    GetCommandName ();

    void
    SetHelp (const char * str);

    void
    SetHelpLong (const char * str);

    void
    SetHelpLong (std::string str);

    void
    SetSyntax (const char *str);
    
    // override this to return true if you want to enable the user to delete
    // the Command object from the Command dictionary (aliases have their own
    // deletion scheme, so they do not need to care about this)
    virtual bool
    IsRemovable() const { return false; }
    
    bool
    IsAlias () { return m_is_alias; }
    
    void
    SetIsAlias (bool value) { m_is_alias = value; }

    virtual bool
    IsMultiwordObject () { return false; }

    virtual lldb::CommandObjectSP
    GetSubcommandSP (const char *sub_cmd, StringList *matches = NULL)
    {
        return lldb::CommandObjectSP();
    }
    
    virtual CommandObject *
    GetSubcommandObject (const char *sub_cmd, StringList *matches = NULL)
    {
        return NULL;
    }
    
    virtual void
    AproposAllSubCommands (const char *prefix,
                           const char *search_word,
                           StringList &commands_found,
                           StringList &commands_help)
    {
    }

    virtual void
    GenerateHelpText (CommandReturnObject &result)
    {
    }

    // this is needed in order to allow the SBCommand class to
    // transparently try and load subcommands - it will fail on
    // anything but a multiword command, but it avoids us doing
    // type checkings and casts
    virtual bool
    LoadSubCommand (const char *cmd_name,
                    const lldb::CommandObjectSP& command_obj)
    {
        return false;
    }
    
    virtual bool
    WantsRawCommandString() = 0;

    // By default, WantsCompletion = !WantsRawCommandString.
    // Subclasses who want raw command string but desire, for example,
    // argument completion should override this method to return true.
    virtual bool
    WantsCompletion() { return !WantsRawCommandString(); }

    virtual Options *
    GetOptions ();

    static const ArgumentTableEntry*
    GetArgumentTable ();

    static lldb::CommandArgumentType
    LookupArgumentName (const char *arg_name);

    static ArgumentTableEntry *
    FindArgumentDataByType (lldb::CommandArgumentType arg_type);

    int
    GetNumArgumentEntries ();

    CommandArgumentEntry *
    GetArgumentEntryAtIndex (int idx);

    static void
    GetArgumentHelp (Stream &str, lldb::CommandArgumentType arg_type, CommandInterpreter &interpreter);

    static const char *
    GetArgumentName (lldb::CommandArgumentType arg_type);

    // Generates a nicely formatted command args string for help command output.
    // By default, all possible args are taken into account, for example,
    // '<expr | variable-name>'.  This can be refined by passing a second arg
    // specifying which option set(s) we are interested, which could then, for
    // example, produce either '<expr>' or '<variable-name>'.
    void
    GetFormattedCommandArguments (Stream &str, uint32_t opt_set_mask = LLDB_OPT_SET_ALL);
    
    bool
    IsPairType (ArgumentRepetitionType arg_repeat_type);
    
    enum
    {
        //----------------------------------------------------------------------
        // eFlagRequiresTarget
        //
        // Ensures a valid target is contained in m_exe_ctx prior to executing
        // the command. If a target doesn't exist or is invalid, the command
        // will fail and CommandObject::GetInvalidTargetDescription() will be
        // returned as the error. CommandObject subclasses can override the
        // virtual function for GetInvalidTargetDescription() to provide custom
        // strings when needed.
        //----------------------------------------------------------------------
        eFlagRequiresTarget         = (1u << 0),
        //----------------------------------------------------------------------
        // eFlagRequiresProcess
        //
        // Ensures a valid process is contained in m_exe_ctx prior to executing
        // the command. If a process doesn't exist or is invalid, the command
        // will fail and CommandObject::GetInvalidProcessDescription() will be
        // returned as the error. CommandObject subclasses can override the
        // virtual function for GetInvalidProcessDescription() to provide custom
        // strings when needed.
        //----------------------------------------------------------------------
        eFlagRequiresProcess        = (1u << 1),
        //----------------------------------------------------------------------
        // eFlagRequiresThread
        //
        // Ensures a valid thread is contained in m_exe_ctx prior to executing
        // the command. If a thread doesn't exist or is invalid, the command
        // will fail and CommandObject::GetInvalidThreadDescription() will be
        // returned as the error. CommandObject subclasses can override the
        // virtual function for GetInvalidThreadDescription() to provide custom
        // strings when needed.
        //----------------------------------------------------------------------
        eFlagRequiresThread         = (1u << 2),
        //----------------------------------------------------------------------
        // eFlagRequiresFrame
        //
        // Ensures a valid frame is contained in m_exe_ctx prior to executing
        // the command. If a frame doesn't exist or is invalid, the command
        // will fail and CommandObject::GetInvalidFrameDescription() will be
        // returned as the error. CommandObject subclasses can override the
        // virtual function for GetInvalidFrameDescription() to provide custom
        // strings when needed.
        //----------------------------------------------------------------------
        eFlagRequiresFrame          = (1u << 3),
        //----------------------------------------------------------------------
        // eFlagRequiresRegContext
        //
        // Ensures a valid register context (from the selected frame if there
        // is a frame in m_exe_ctx, or from the selected thread from m_exe_ctx)
        // is availble from m_exe_ctx prior to executing the command. If a
        // target doesn't exist or is invalid, the command will fail and
        // CommandObject::GetInvalidRegContextDescription() will be returned as
        // the error. CommandObject subclasses can override the virtual function
        // for GetInvalidRegContextDescription() to provide custom strings when
        // needed.
        //----------------------------------------------------------------------
        eFlagRequiresRegContext     = (1u << 4),
        //----------------------------------------------------------------------
        // eFlagTryTargetAPILock
        //
        // Attempts to acquire the target lock if a target is selected in the
        // command interpreter. If the command object fails to acquire the API
        // lock, the command will fail with an appropriate error message.
        //----------------------------------------------------------------------
        eFlagTryTargetAPILock       = (1u << 5),
        //----------------------------------------------------------------------
        // eFlagProcessMustBeLaunched
        //
        // Verifies that there is a launched process in m_exe_ctx, if there
        // isn't, the command will fail with an appropriate error message.
        //----------------------------------------------------------------------
        eFlagProcessMustBeLaunched  = (1u << 6),
        //----------------------------------------------------------------------
        // eFlagProcessMustBePaused
        //
        // Verifies that there is a paused process in m_exe_ctx, if there
        // isn't, the command will fail with an appropriate error message.
        //----------------------------------------------------------------------
        eFlagProcessMustBePaused    = (1u << 7)
    };

    bool
    ParseOptions (Args& args, CommandReturnObject &result);

    void
    SetCommandName (const char *name);

    // This function really deals with CommandObjectLists, but we didn't make a
    // CommandObjectList class, so I'm sticking it here.  But we really should have
    // such a class.  Anyway, it looks up the commands in the map that match the partial
    // string cmd_str, inserts the matches into matches, and returns the number added.

    static int
    AddNamesMatchingPartialString (CommandMap &in_map, const char *cmd_str, StringList &matches);

    //------------------------------------------------------------------
    /// The input array contains a parsed version of the line.  The insertion
    /// point is given by cursor_index (the index in input of the word containing
    /// the cursor) and cursor_char_position (the position of the cursor in that word.)
    /// This default version handles calling option argument completions and then calls
    /// HandleArgumentCompletion if the cursor is on an argument, not an option.
    /// Don't override this method, override HandleArgumentCompletion instead unless
    /// you have special reasons.
    ///
    /// @param[in] interpreter
    ///    The command interpreter doing the completion.
    ///
    /// @param[in] input
    ///    The command line parsed into words
    ///
    /// @param[in] cursor_index
    ///     The index in \ainput of the word in which the cursor lies.
    ///
    /// @param[in] cursor_char_pos
    ///     The character position of the cursor in its argument word.
    ///
    /// @param[in] match_start_point
    /// @param[in] match_return_elements
    ///     FIXME: Not yet implemented...  If there is a match that is expensive to compute, these are
    ///     here to allow you to compute the completions in batches.  Start the completion from \amatch_start_point,
    ///     and return \amatch_return_elements elements.
    ///
    /// @param[out] word_complete
    ///     \btrue if this is a complete option value (a space will be inserted after the
    ///     completion.)  \bfalse otherwise.
    ///
    /// @param[out] matches
    ///     The array of matches returned.
    ///
    /// FIXME: This is the wrong return value, since we also need to make a distinction between
    /// total number of matches, and the window the user wants returned.
    ///
    /// @return
    ///     \btrue if we were in an option, \bfalse otherwise.
    //------------------------------------------------------------------
    virtual int
    HandleCompletion (Args &input,
                      int &cursor_index,
                      int &cursor_char_position,
                      int match_start_point,
                      int max_return_elements,
                      bool &word_complete,
                      StringList &matches);

    //------------------------------------------------------------------
    /// The input array contains a parsed version of the line.  The insertion
    /// point is given by cursor_index (the index in input of the word containing
    /// the cursor) and cursor_char_position (the position of the cursor in that word.)
    /// We've constructed the map of options and their arguments as well if that is
    /// helpful for the completion.
    ///
    /// @param[in] interpreter
    ///    The command interpreter doing the completion.
    ///
    /// @param[in] input
    ///    The command line parsed into words
    ///
    /// @param[in] cursor_index
    ///     The index in \ainput of the word in which the cursor lies.
    ///
    /// @param[in] cursor_char_pos
    ///     The character position of the cursor in its argument word.
    ///
    /// @param[in] opt_element_vector
    ///     The results of the options parse of \a input.
    ///
    /// @param[in] match_start_point
    /// @param[in] match_return_elements
    ///     See CommandObject::HandleCompletions for a description of how these work.
    ///
    /// @param[out] word_complete
    ///     \btrue if this is a complete option value (a space will be inserted after the
    ///     completion.)  \bfalse otherwise.
    ///
    /// @param[out] matches
    ///     The array of matches returned.
    ///
    /// FIXME: This is the wrong return value, since we also need to make a distinction between
    /// total number of matches, and the window the user wants returned.
    ///
    /// @return
    ///     The number of completions.
    //------------------------------------------------------------------

    virtual int
    HandleArgumentCompletion (Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches)
    {
        return 0;
    }
    
    bool
    HelpTextContainsWord (const char *search_word);

    //------------------------------------------------------------------
    /// The flags accessor.
    ///
    /// @return
    ///     A reference to the Flags member variable.
    //------------------------------------------------------------------
    Flags&
    GetFlags()
    {
        return m_flags;
    }

    //------------------------------------------------------------------
    /// The flags const accessor.
    ///
    /// @return
    ///     A const reference to the Flags member variable.
    //------------------------------------------------------------------
    const Flags&
    GetFlags() const
    {
        return m_flags;
    }
    
    //------------------------------------------------------------------
    /// Get the command that appropriate for a "repeat" of the current command.
    ///
    /// @param[in] current_command_line
    ///    The complete current command line.
    ///
    /// @return
    ///     NULL if there is no special repeat command - it will use the current command line.
    ///     Otherwise a pointer to the command to be repeated.
    ///     If the returned string is the empty string, the command won't be repeated.    
    //------------------------------------------------------------------
    virtual const char *GetRepeatCommand (Args &current_command_args, uint32_t index)
    {
        return NULL;
    }

    CommandOverrideCallback
    GetOverrideCallback () const
    {
        return m_command_override_callback;
    }
    
    void *
    GetOverrideCallbackBaton () const
    {
        return m_command_override_baton;
    }

    void
    SetOverrideCallback (CommandOverrideCallback callback, void *baton)
    {
        m_command_override_callback = callback;
        m_command_override_baton = baton;
    }
    
    virtual bool
    Execute (const char *args_string, CommandReturnObject &result) = 0;

protected:
    virtual const char *
    GetInvalidTargetDescription()
    {
        return "invalid target, create a target using the 'target create' command";
    }

    virtual const char *
    GetInvalidProcessDescription()
    {
        return "invalid process";
    }

    virtual const char *
    GetInvalidThreadDescription()
    {
        return "invalid thread";
    }
    
    virtual const char *
    GetInvalidFrameDescription()
    {
        return "invalid frame";
    }
    
    virtual const char *
    GetInvalidRegContextDescription ()
    {
        return "invalid frame, no registers";
    }

    //------------------------------------------------------------------
    /// Check the command to make sure anything required by this
    /// command is available.
    ///
    /// @param[out] result
    ///     A command result object, if it is not okay to run the command
    ///     this will be filled in with a suitable error.
    ///
    /// @return
    ///     \b true if it is okay to run this command, \b false otherwise.
    //------------------------------------------------------------------
    bool
    CheckRequirements (CommandReturnObject &result);
    
    void
    Cleanup ();

    CommandInterpreter &m_interpreter;
    ExecutionContext m_exe_ctx;
    Mutex::Locker m_api_locker;
    std::string m_cmd_name;
    std::string m_cmd_help_short;
    std::string m_cmd_help_long;
    std::string m_cmd_syntax;
    bool m_is_alias;
    Flags m_flags;
    std::vector<CommandArgumentEntry> m_arguments;
    CommandOverrideCallback m_command_override_callback;
    void * m_command_override_baton;
    
    // Helper function to populate IDs or ID ranges as the command argument data
    // to the specified command argument entry.
    static void
    AddIDsArgumentData(CommandArgumentEntry &arg, lldb::CommandArgumentType ID, lldb::CommandArgumentType IDRange);
    
};

class CommandObjectParsed : public CommandObject
{
public:

    CommandObjectParsed (CommandInterpreter &interpreter,
                         const char *name,
                         const char *help = NULL,
                         const char *syntax = NULL,
                         uint32_t flags = 0) :
        CommandObject (interpreter, name, help, syntax, flags) {}

    virtual
    ~CommandObjectParsed () {};
    
    virtual bool
    Execute (const char *args_string, CommandReturnObject &result);
    
protected:
    virtual bool
    DoExecute (Args& command,
             CommandReturnObject &result) = 0;
    
    virtual bool
    WantsRawCommandString() { return false; };
};

class CommandObjectRaw : public CommandObject
{
public:

    CommandObjectRaw (CommandInterpreter &interpreter,
                         const char *name,
                         const char *help = NULL,
                         const char *syntax = NULL,
                         uint32_t flags = 0) :
        CommandObject (interpreter, name, help, syntax, flags) {}

    virtual
    ~CommandObjectRaw () {};
    
    virtual bool
    Execute (const char *args_string, CommandReturnObject &result);
    
protected:    
    virtual bool
    DoExecute (const char *command, CommandReturnObject &result) = 0;

    virtual bool
    WantsRawCommandString() { return true; };
};


} // namespace lldb_private


#endif  // liblldb_CommandObject_h_
