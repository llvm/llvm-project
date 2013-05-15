//===-- OptionValueFileSpec.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_OptionValueFileSpec_h_
#define liblldb_OptionValueFileSpec_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Host/FileSpec.h"
#include "lldb/Interpreter/OptionValue.h"

namespace lldb_private {

class OptionValueFileSpec : public OptionValue
{
public:
    OptionValueFileSpec ();
    
    OptionValueFileSpec (const FileSpec &value);
    
    OptionValueFileSpec (const FileSpec &current_value, 
                         const FileSpec &default_value);
    
    virtual 
    ~OptionValueFileSpec()
    {
    }
    
    //---------------------------------------------------------------------
    // Virtual subclass pure virtual overrides
    //---------------------------------------------------------------------
    
    virtual OptionValue::Type
    GetType () const
    {
        return eTypeFileSpec;
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
        m_data_sp.reset();
        return true;
    }
    
    virtual lldb::OptionValueSP
    DeepCopy () const;

    virtual size_t
    AutoComplete (CommandInterpreter &interpreter,
                  const char *s,
                  int match_start_point,
                  int max_return_elements,
                  bool &word_complete,
                  StringList &matches);
    
    //---------------------------------------------------------------------
    // Subclass specific functions
    //---------------------------------------------------------------------
    
    FileSpec &
    GetCurrentValue()
    {
        return m_current_value;
    }

    const FileSpec &
    GetCurrentValue() const
    {
        return m_current_value;
    }

    const FileSpec &
    GetDefaultValue() const
    {
        return m_default_value;
    }
    
    void
    SetCurrentValue (const FileSpec &value, bool set_value_was_set)
    {
        m_current_value = value;
        if (set_value_was_set)
            m_value_was_set = true;
        m_data_sp.reset();
    }
    
    void
    SetDefaultValue (const FileSpec &value)
    {
        m_default_value = value;
    }
    
    const lldb::DataBufferSP &
    GetFileContents(bool null_terminate);
    
    void
    SetCompletionMask (uint32_t mask)
    {
        m_completion_mask = mask;
    }

protected:
    FileSpec m_current_value;
    FileSpec m_default_value;
    lldb::DataBufferSP m_data_sp;
    uint32_t m_completion_mask;
};

} // namespace lldb_private

#endif  // liblldb_OptionValueFileSpec_h_
