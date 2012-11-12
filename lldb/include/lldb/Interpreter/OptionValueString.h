//===-- OptionValueString.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_OptionValueString_h_
#define liblldb_OptionValueString_h_

// C Includes
// C++ Includes
#include <string>

// Other libraries and framework includes
// Project includes
#include "lldb/Core/Flags.h"
#include "lldb/Interpreter/OptionValue.h"

namespace lldb_private {

class OptionValueString : public OptionValue
{
public:
    enum Options
    {
        eOptionEncodeCharacterEscapeSequences = (1u << 0)
    };

    OptionValueString () :
        OptionValue(),
        m_current_value (),
        m_default_value (),
        m_options()
    {
    }

    OptionValueString (const char *value) :
        OptionValue(),
        m_current_value (),
        m_default_value (),
        m_options()
    {
        if  (value && value[0])
        {
            m_current_value.assign (value);
            m_default_value.assign (value);
        }
    }

    OptionValueString (const char *current_value,
                       const char *default_value) :
        OptionValue(),
        m_current_value (),
        m_default_value (),
        m_options()
    {
        if  (current_value && current_value[0])
            m_current_value.assign (current_value);
        if  (default_value && default_value[0])
            m_default_value.assign (default_value);
    }
    
    virtual 
    ~OptionValueString()
    {
    }
    
    //---------------------------------------------------------------------
    // Virtual subclass pure virtual overrides
    //---------------------------------------------------------------------
    
    virtual OptionValue::Type
    GetType () const
    {
        return eTypeString;
    }
    
    virtual void
    DumpValue (const ExecutionContext *exe_ctx, Stream &strm, uint32_t dump_mask);
    
    virtual Error
    SetValueFromCString (const char *value,
                         VarSetOperationType op = eVarSetOperationAssign);

    virtual bool
    Clear ()
    {
        m_current_value = m_default_value;
        m_value_was_set = false;
        return true;
    }

    virtual lldb::OptionValueSP
    DeepCopy () const;

    //---------------------------------------------------------------------
    // Subclass specific functions
    //---------------------------------------------------------------------

    Flags &
    GetOptions ()
    {
        return m_options;
    }

    const Flags &
    GetOptions () const
    {
        return m_options;
    }
    
    const char *
    operator = (const char *value)
    {
        if (value && value[0])
            m_current_value.assign (value);
        else
            m_current_value.clear();
        return m_current_value.c_str();
    }

    const char *
    GetCurrentValue() const
    {
        return m_current_value.c_str();
    }
    
    const char *
    GetDefaultValue() const
    {
        return m_default_value.c_str();
    }
    
    void
    SetCurrentValue (const char *value)
    {
        if (value && value[0])
            m_current_value.assign (value);
        else
            m_current_value.clear();
    }

    void
    AppendToCurrentValue (const char *value)
    {
        if (value && value[0])
            m_current_value.append (value);
    }

    void
    SetDefaultValue (const char *value)
    {
        if (value && value[0])
            m_default_value.assign (value);
        else
            m_default_value.clear();
    }

    bool
    IsCurrentValueEmpty () const
    {
        return m_current_value.empty();
    }

    bool
    IsDefaultValueEmpty () const
    {
        return m_default_value.empty();
    }

    
protected:
    std::string m_current_value;
    std::string m_default_value;
    Flags m_options;
};

} // namespace lldb_private

#endif  // liblldb_OptionValueString_h_
