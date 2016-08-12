//===-- OptionGroupFile.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_OptionGroupFile_h_
#define liblldb_OptionGroupFile_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/Options.h"
#include "lldb/Interpreter/OptionValueFileSpec.h"
#include "lldb/Interpreter/OptionValueFileSpecList.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// OptionGroupFile
//-------------------------------------------------------------------------

class OptionGroupFile : public OptionGroup
{
public:
    OptionGroupFile (uint32_t usage_mask,
                     bool required,
                     const char *long_option, 
                     int short_option,
                     uint32_t completion_type,
                     lldb::CommandArgumentType argument_type,
                     const char *usage_text);
    
    ~OptionGroupFile() override;

    uint32_t
    GetNumDefinitions() override
    {
        return 1;
    }
    
    const OptionDefinition*
    GetDefinitions() override
    {
        return &m_option_definition;
    }
    
    Error
    SetOptionValue(uint32_t option_idx,
                   const char *option_value,
                   ExecutionContext *execution_context) override;
    
    void
    OptionParsingStarting(ExecutionContext *execution_context) override;
    
    OptionValueFileSpec &
    GetOptionValue ()
    {
        return m_file;
    }

    const OptionValueFileSpec &
    GetOptionValue () const
    {
        return m_file;
    }

protected:
    OptionValueFileSpec m_file;
    OptionDefinition m_option_definition;
};

//-------------------------------------------------------------------------
// OptionGroupFileList
//-------------------------------------------------------------------------

class OptionGroupFileList : public OptionGroup
{
public:
    
    OptionGroupFileList (uint32_t usage_mask,
                         bool required,
                         const char *long_option, 
                         int short_option,
                         uint32_t completion_type,
                         lldb::CommandArgumentType argument_type,
                         const char *usage_text);
    
    ~OptionGroupFileList() override;
    
    uint32_t
    GetNumDefinitions() override
    {
        return 1;
    }
    
    const OptionDefinition*
    GetDefinitions() override
    {
        return &m_option_definition;
    }
    
    Error
    SetOptionValue(uint32_t option_idx,
                   const char *option_value,
                   ExecutionContext *execution_context) override;
    
    void
    OptionParsingStarting(ExecutionContext *execution_context) override;

    OptionValueFileSpecList &
    GetOptionValue ()
    {
        return m_file_list;
    }
    
    const OptionValueFileSpecList &
    GetOptionValue () const
    {
        return m_file_list;
    }
    
protected:
    OptionValueFileSpecList m_file_list;
    OptionDefinition m_option_definition;
};

} // namespace lldb_private

#endif // liblldb_OptionGroupFile_h_
