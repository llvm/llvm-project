//===-- OptionValueRegex.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_OptionValueRegex_h_
#define liblldb_OptionValueRegex_h_

// C Includes
// C++ Includes
#include <string>

// Other libraries and framework includes
// Project includes
#include "lldb/Core/RegularExpression.h"
#include "lldb/Interpreter/OptionValue.h"

namespace lldb_private {

class OptionValueRegex : public OptionValue
{
public:
    OptionValueRegex (const char *value = NULL, uint32_t regex_flags = 0) :
        OptionValue(),
        m_regex (value, regex_flags)
    {
    }

    virtual 
    ~OptionValueRegex()
    {
    }
    
    //---------------------------------------------------------------------
    // Virtual subclass pure virtual overrides
    //---------------------------------------------------------------------
    
    virtual OptionValue::Type
    GetType () const
    {
        return eTypeRegex;
    }
    
    virtual void
    DumpValue (const ExecutionContext *exe_ctx, Stream &strm, uint32_t dump_mask);
    
    virtual Error
    SetValueFromCString (const char *value,
                         VarSetOperationType op = eVarSetOperationAssign);

    virtual bool
    Clear ()
    {
        m_regex.Clear();
        m_value_was_set = false;
        return true;
    }

    virtual lldb::OptionValueSP
    DeepCopy () const;

    //---------------------------------------------------------------------
    // Subclass specific functions
    //---------------------------------------------------------------------
    const RegularExpression *
    GetCurrentValue() const
    {
        if (m_regex.IsValid())
            return &m_regex;
        return NULL;
    }
    
    void
    SetCurrentValue (const char *value, uint32_t regex_flags)
    {
        if (value && value[0])
            m_regex.Compile (value, regex_flags);
        else
            m_regex.Clear();
    }

    bool
    IsValid () const
    {
        return m_regex.IsValid();
    }
    
protected:
    RegularExpression m_regex;
};

} // namespace lldb_private

#endif  // liblldb_OptionValueRegex_h_
