//===-- CommandObjectArgs.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectArgs_h_
#define liblldb_CommandObjectArgs_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Core/Language.h"

namespace lldb_private {
    
    class CommandObjectArgs : public CommandObjectParsed
    {
    public:
        
        class CommandOptions : public Options
        {
        public:
            
            CommandOptions (CommandInterpreter &interpreter);
            
            virtual
            ~CommandOptions ();
            
            virtual Error
            SetOptionValue (uint32_t option_idx, const char *option_arg);
            
            void
            OptionParsingStarting ();
            
            const OptionDefinition*
            GetDefinitions ();
            
            // Options table: Required for subclasses of Options.
            
            static OptionDefinition g_option_table[];
        };
        
        CommandObjectArgs (CommandInterpreter &interpreter);
        
        virtual
        ~CommandObjectArgs ();
        
        virtual
        Options *
        GetOptions ();
        
        
    protected:
        
        CommandOptions m_options;

        virtual bool
        DoExecute (    Args& command,
                 CommandReturnObject &result);
        
    };
    
} // namespace lldb_private

#endif  // liblldb_CommandObjectArgs_h_
