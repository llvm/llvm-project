//===-- OptionGroupArchitecture.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_OptionGroupArchitecture_h_
#define liblldb_OptionGroupArchitecture_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/Options.h"
#include "lldb/Core/ArchSpec.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// OptionGroupArchitecture
//-------------------------------------------------------------------------

class OptionGroupArchitecture : public OptionGroup
{
public:
    
    OptionGroupArchitecture ();
    
    virtual
    ~OptionGroupArchitecture ();

    
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
    
    bool
    GetArchitecture (Platform *platform, ArchSpec &arch);

    bool
    ArchitectureWasSpecified () const
    {
        return !m_arch_str.empty();
    }
    const char *
    GetArchitectureName ()
    {
        if (m_arch_str.empty())
            return NULL;
        return m_arch_str.c_str();
    }

protected:

    std::string m_arch_str; // Save the arch triple in case a platform is specified after the architecture
};

} // namespace lldb_private

#endif  // liblldb_OptionGroupArchitecture_h_
