//===-- CommandObjectHelp.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectHelp_h_
#define liblldb_CommandObjectHelp_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Interpreter/Options.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectHelp
//-------------------------------------------------------------------------

class CommandObjectHelp : public CommandObjectParsed
{
public:

    CommandObjectHelp (CommandInterpreter &interpreter);

    virtual
    ~CommandObjectHelp ();

    virtual int
    HandleCompletion (Args &input,
                      int &cursor_index,
                      int &cursor_char_position,
                      int match_start_point,
                      int max_return_elements,
                      bool &word_complete,
                      StringList &matches);
    
    class CommandOptions : public Options
    {
    public:
        
        CommandOptions (CommandInterpreter &interpreter) :
        Options (interpreter)
        {
        }
        
        virtual
        ~CommandOptions (){}
        
        virtual Error
        SetOptionValue (uint32_t option_idx, const char *option_arg)
        {
            Error error;
            const int short_option = m_getopt_table[option_idx].val;
            
            switch (short_option)
            {
                case 'a':
                    m_show_aliases = true;
                    break;
                case 'u':
                    m_show_user_defined = false;
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
            m_show_aliases = false;
            m_show_user_defined = true;
        }
        
        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }
        
        // Options table: Required for subclasses of Options.
        
        static OptionDefinition g_option_table[];
        
        // Instance variables to hold the values for command options.
        
        bool m_show_aliases;
        bool m_show_user_defined;        
    };
    
    virtual Options *
    GetOptions ()
    {
        return &m_options;
    }
    
protected:
    virtual bool
    DoExecute (Args& command,
             CommandReturnObject &result);

private:
    CommandOptions m_options;
    
};

} // namespace lldb_private

#endif  // liblldb_CommandObjectHelp_h_
