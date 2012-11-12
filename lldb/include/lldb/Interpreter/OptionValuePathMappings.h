//===-- OptionValuePathMappings.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_OptionValuePathMappings_h_
#define liblldb_OptionValuePathMappings_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Target/PathMappingList.h"
#include "lldb/Interpreter/OptionValue.h"

namespace lldb_private {

class OptionValuePathMappings : public OptionValue
{
public:
    OptionValuePathMappings (bool notify_changes) :
        OptionValue(),
        m_path_mappings (),
        m_notify_changes (notify_changes)
    {
    }
    
    virtual 
    ~OptionValuePathMappings()
    {
    }
    
    //---------------------------------------------------------------------
    // Virtual subclass pure virtual overrides
    //---------------------------------------------------------------------
    
    virtual OptionValue::Type
    GetType () const
    {
        return eTypePathMap;
    }
    
    virtual void
    DumpValue (const ExecutionContext *exe_ctx, Stream &strm, uint32_t dump_mask);
    
    virtual Error
    SetValueFromCString (const char *value,
                         VarSetOperationType op = eVarSetOperationAssign);
    
    virtual bool
    Clear ()
    {
        m_path_mappings.Clear(m_notify_changes);
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
    
    PathMappingList &
    GetCurrentValue()
    {
        return m_path_mappings;
    }
    
    const PathMappingList &
    GetCurrentValue() const
    {
        return m_path_mappings;
    }
    
protected:
    PathMappingList m_path_mappings;
    bool m_notify_changes;
};

} // namespace lldb_private

#endif  // liblldb_OptionValuePathMappings_h_
