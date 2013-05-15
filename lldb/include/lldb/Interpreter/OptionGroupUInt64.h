//===-- OptionGroupUInt64.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_OptionGroupUInt64_h_
#define liblldb_OptionGroupUInt64_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/Options.h"
#include "lldb/Interpreter/OptionValueUInt64.h"

namespace lldb_private {
    //-------------------------------------------------------------------------
    // OptionGroupUInt64
    //-------------------------------------------------------------------------
    
    class OptionGroupUInt64 : public OptionGroup
    {
    public:
        
        OptionGroupUInt64 (uint32_t usage_mask,
                           bool required,
                           const char *long_option, 
                           int short_option,
                           uint32_t completion_type,
                           lldb::CommandArgumentType argument_type,
                           const char *usage_text,
                           uint64_t default_value);
        
        virtual
        ~OptionGroupUInt64 ();
        
        
        virtual uint32_t
        GetNumDefinitions ()
        {
            return 1;
        }
        
        virtual const OptionDefinition*
        GetDefinitions ()
        {
            return &m_option_definition;
        }
        
        virtual Error
        SetOptionValue (CommandInterpreter &interpreter,
                        uint32_t option_idx,
                        const char *option_value);
        
        virtual void
        OptionParsingStarting (CommandInterpreter &interpreter);
        
        OptionValueUInt64 &
        GetOptionValue ()
        {
            return m_value;
        }
        
        const OptionValueUInt64 &
        GetOptionValue () const
        {
            return m_value;
        }
        
    protected:
        OptionValueUInt64 m_value;
        OptionDefinition m_option_definition;
        
    };
    
} // namespace lldb_private

#endif  // liblldb_OptionGroupUInt64_h_
