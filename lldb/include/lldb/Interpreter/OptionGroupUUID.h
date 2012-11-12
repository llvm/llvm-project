//===-- OptionGroupUUID.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_OptionGroupUUID_h_
#define liblldb_OptionGroupUUID_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/Options.h"
#include "lldb/Interpreter/OptionValueUUID.h"

namespace lldb_private {
//-------------------------------------------------------------------------
// OptionGroupUUID
//-------------------------------------------------------------------------

class OptionGroupUUID : public OptionGroup
{
public:
    
    OptionGroupUUID ();
    
    virtual
    ~OptionGroupUUID ();

    
    virtual uint32_t
    GetNumDefinitions ();
    
    virtual const OptionDefinition*
    GetDefinitions ();
    
    virtual Error
    SetOptionValue (CommandInterpreter &interpreter,
                    uint32_t option_idx,
                    const char *option_value);
    
    virtual void
    OptionParsingStarting (CommandInterpreter &interpreter);
    
    const OptionValueUUID &
    GetOptionValue () const
    {
        return m_uuid;
    }

protected:
    OptionValueUUID m_uuid;
};

} // namespace lldb_private

#endif  // liblldb_OptionGroupUUID_h_
