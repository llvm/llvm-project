//===-- OptionValueArgs.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_OptionValueArgs_h_
#define liblldb_OptionValueArgs_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/OptionValueArray.h"

namespace lldb_private {

class OptionValueArgs : public OptionValueArray
{
public:
    OptionValueArgs () :
        OptionValueArray (OptionValue::ConvertTypeToMask (OptionValue::eTypeString))
    {
    }
    
    virtual
    ~OptionValueArgs()
    {
    }
    
    size_t
    GetArgs (Args &args);
    
    virtual Type
    GetType() const
    {
        return eTypeArgs;
    }
};

} // namespace lldb_private

#endif  // liblldb_OptionValueArgs_h_
