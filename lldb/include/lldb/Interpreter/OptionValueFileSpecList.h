//===-- OptionValueFileSpecList.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_OptionValueFileSpecList_h_
#define liblldb_OptionValueFileSpecList_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/FileSpecList.h"
#include "lldb/Interpreter/OptionValue.h"

namespace lldb_private {

class OptionValueFileSpecList : public OptionValue
{
public:
    OptionValueFileSpecList () :
        OptionValue(),
        m_current_value ()
    {
    }
    
    OptionValueFileSpecList (const FileSpecList &current_value) :
        OptionValue(),
        m_current_value (current_value)
    {
    }
    
    
    virtual 
    ~OptionValueFileSpecList()
    {
    }
    
    //---------------------------------------------------------------------
    // Virtual subclass pure virtual overrides
    //---------------------------------------------------------------------
    
    virtual OptionValue::Type
    GetType () const
    {
        return eTypeFileSpecList;
    }
    
    virtual void
    DumpValue (const ExecutionContext *exe_ctx, Stream &strm, uint32_t dump_mask);
    
    virtual Error
    SetValueFromCString (const char *value,
                         VarSetOperationType op = eVarSetOperationAssign);
    
    virtual bool
    Clear ()
    {
        m_current_value.Clear();
        m_value_was_set = false;
        return true;
    }
    
    virtual lldb::OptionValueSP
    DeepCopy () const;
    
    virtual bool
    IsAggregateValue () const
    {
        return true;
    }

    //---------------------------------------------------------------------
    // Subclass specific functions
    //---------------------------------------------------------------------
    
    FileSpecList &
    GetCurrentValue()
    {
        return m_current_value;
    }
    
    const FileSpecList &
    GetCurrentValue() const
    {
        return m_current_value;
    }
    
    void
    SetCurrentValue (const FileSpecList &value)
    {
        m_current_value = value;
    }
    
protected:
    FileSpecList m_current_value;
};

} // namespace lldb_private

#endif  // liblldb_OptionValueFileSpecList_h_
