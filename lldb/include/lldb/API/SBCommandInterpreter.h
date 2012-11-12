//===-- SBCommandInterpreter.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBCommandInterpreter_h_
#define LLDB_SBCommandInterpreter_h_

#include "lldb/API/SBDefines.h"
#include "lldb/API/SBDebugger.h"

namespace lldb {

class SBCommandInterpreter
{
public:
    enum
    {
        eBroadcastBitThreadShouldExit       = (1 << 0),
        eBroadcastBitResetPrompt            = (1 << 1),
        eBroadcastBitQuitCommandReceived    = (1 << 2),           // User entered quit 
        eBroadcastBitAsynchronousOutputData = (1 << 3),
        eBroadcastBitAsynchronousErrorData  = (1 << 4)
    };

    SBCommandInterpreter (const lldb::SBCommandInterpreter &rhs);
    
    const lldb::SBCommandInterpreter &
    operator = (const lldb::SBCommandInterpreter &rhs);

    ~SBCommandInterpreter ();

    static const char * 
    GetArgumentTypeAsCString (const lldb::CommandArgumentType arg_type);
    
    static const char *
    GetArgumentDescriptionAsCString (const lldb::CommandArgumentType arg_type);
    
    bool
    IsValid() const;

    bool
    CommandExists (const char *cmd);

    bool
    AliasExists (const char *cmd);

    lldb::SBBroadcaster
    GetBroadcaster ();
    
    static const char *
    GetBroadcasterClass ();

    bool
    HasCommands ();

    bool
    HasAliases ();

    bool
    HasAliasOptions ();

    lldb::SBProcess
    GetProcess ();
    
    lldb::SBDebugger
    GetDebugger ();
    
    lldb::SBCommand
    AddMultiwordCommand (const char* name, const char* help);
    
    lldb::SBCommand
    AddCommand (const char* name, lldb::SBCommandPluginInterface *impl, const char* help);

    void
    SourceInitFileInHomeDirectory (lldb::SBCommandReturnObject &result);

    void
    SourceInitFileInCurrentWorkingDirectory (lldb::SBCommandReturnObject &result);

    lldb::ReturnStatus
    HandleCommand (const char *command_line, lldb::SBCommandReturnObject &result, bool add_to_history = false);

    // This interface is not useful in SWIG, since the cursor & last_char arguments are string pointers INTO current_line
    // and you can't do that in a scripting language interface in general... 
    int
    HandleCompletion (const char *current_line,
                      const char *cursor,
                      const char *last_char,
                      int match_start_point,
                      int max_return_elements,
                      lldb::SBStringList &matches);

    int
    HandleCompletion (const char *current_line,
                      uint32_t cursor_pos,
                      int match_start_point,
                      int max_return_elements,
                      lldb::SBStringList &matches);

    // Catch commands before they execute by registering a callback that will
    // get called when the command gets executed. This allows GUI or command
    // line interfaces to intercept a command and stop it from happening
    bool
    SetCommandOverrideCallback (const char *command_name,
                                lldb::CommandOverrideCallback callback,
                                void *baton);
    
    SBCommandInterpreter (lldb_private::CommandInterpreter *interpreter_ptr = NULL);   // Access using SBDebugger::GetCommandInterpreter();
    
protected:

    lldb_private::CommandInterpreter &
    ref ();

    lldb_private::CommandInterpreter *
    get ();

    void
    reset (lldb_private::CommandInterpreter *);
private:
    friend class SBDebugger;

    static void
    InitializeSWIG ();

    lldb_private::CommandInterpreter *m_opaque_ptr;
};

class SBCommandPluginInterface
{
public:
    virtual bool
    DoExecute (lldb::SBDebugger debugger,
               char** command,
               lldb::SBCommandReturnObject &result)
    {
        return false;
    }
    
    virtual
    ~SBCommandPluginInterface ()
    {}
};
    
class SBCommand
{
public:
    
    SBCommand ();
    
    bool
    IsValid ();
    
    const char*
    GetName ();
    
    const char*
    GetHelp ();
    
    lldb::SBCommand
    AddMultiwordCommand (const char* name, const char* help = NULL);
    
    lldb::SBCommand
    AddCommand (const char* name, lldb::SBCommandPluginInterface* impl, const char* help = NULL);
    
private:
    
    friend class SBDebugger;
    friend class SBCommandInterpreter;
    
    SBCommand (lldb::CommandObjectSP cmd_sp);
    
    lldb::CommandObjectSP m_opaque_sp;
};

} // namespace lldb

#endif // LLDB_SBCommandInterpreter_h_
