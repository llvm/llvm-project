//===-- OptionValueDictionary.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_OptionValueDictionary_h_
#define liblldb_OptionValueDictionary_h_

// C Includes
// C++ Includes
#include <map>

// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/OptionValue.h"

namespace lldb_private {
    
class OptionValueDictionary : public OptionValue
{
public:
    OptionValueDictionary (uint32_t type_mask = UINT32_MAX, bool raw_value_dump = true) :
        OptionValue(),
        m_type_mask (type_mask),
        m_values (),
        m_raw_value_dump (raw_value_dump)
    {
    }
    
    virtual 
    ~OptionValueDictionary()
    {
    }
    
    //---------------------------------------------------------------------
    // Virtual subclass pure virtual overrides
    //---------------------------------------------------------------------
    
    virtual OptionValue::Type
    GetType () const
    {
        return eTypeDictionary;
    }
    
    virtual void
    DumpValue (const ExecutionContext *exe_ctx, Stream &strm, uint32_t dump_mask);
    
    virtual Error
    SetValueFromCString (const char *value,
                         VarSetOperationType op = eVarSetOperationAssign);
    
    virtual bool
    Clear ()
    {
        m_values.clear();
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

    bool
    IsHomogenous() const
    {
        return ConvertTypeMaskToType (m_type_mask) != eTypeInvalid;
    }

    //---------------------------------------------------------------------
    // Subclass specific functions
    //---------------------------------------------------------------------
    
    size_t
    GetNumValues() const
    {
        return m_values.size();
    }
    
    lldb::OptionValueSP
    GetValueForKey (const ConstString &key) const;
    
    virtual lldb::OptionValueSP
    GetSubValue (const ExecutionContext *exe_ctx,
                 const char *name,
                 bool will_modify,
                 Error &error) const;
    
    virtual Error
    SetSubValue (const ExecutionContext *exe_ctx,
                 VarSetOperationType op,
                 const char *name,
                 const char *value);

    //---------------------------------------------------------------------
    // String value getters and setters
    //---------------------------------------------------------------------
    const char *
    GetStringValueForKey (const ConstString &key);

    bool
    SetStringValueForKey (const ConstString &key, 
                          const char *value,
                          bool can_replace = true);

    
    bool
    SetValueForKey (const ConstString &key, 
                    const lldb::OptionValueSP &value_sp, 
                    bool can_replace = true);
    
    bool
    DeleteValueForKey (const ConstString &key);
    
    size_t
    GetArgs (Args &args) const;
    
    Error
    SetArgs (const Args &args, VarSetOperationType op);
    
protected:
    typedef std::map<ConstString, lldb::OptionValueSP> collection;
    uint32_t m_type_mask;
    collection m_values;
    bool m_raw_value_dump;
};
    
} // namespace lldb_private

#endif  // liblldb_OptionValueDictionary_h_
