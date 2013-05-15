//===-- SBEvent.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBExpressionOptions_h_
#define LLDB_SBExpressionOptions_h_

#include "lldb/API/SBDefines.h"

#include <vector>

namespace lldb {


class SBExpressionOptions
{
public:
    SBExpressionOptions();

    SBExpressionOptions (const lldb::SBExpressionOptions &rhs);
    
    ~SBExpressionOptions();

    const SBExpressionOptions &
    operator = (const lldb::SBExpressionOptions &rhs);

    bool
    GetCoerceResultToId () const;
    
    void
    SetCoerceResultToId (bool coerce = true);
    
    bool
    GetUnwindOnError () const;
    
    void
    SetUnwindOnError (bool unwind = true);
    
    bool
    GetIgnoreBreakpoints () const;
    
    void
    SetIgnoreBreakpoints (bool ignore = true);
    
    lldb::DynamicValueType
    GetFetchDynamicValue () const;
    
    void
    SetFetchDynamicValue (lldb::DynamicValueType dynamic = lldb::eDynamicCanRunTarget);
    
    uint32_t
    GetTimeoutInMicroSeconds () const;
    
    void
    SetTimeoutInMicroSeconds (uint32_t timeout = 0);
    
    bool
    GetTryAllThreads () const;
    
    void
    SetTryAllThreads (bool run_others = true);

protected:

    SBExpressionOptions (lldb_private::EvaluateExpressionOptions &expression_options);

    lldb_private::EvaluateExpressionOptions *
    get () const;

    lldb_private::EvaluateExpressionOptions &
    ref () const;

    friend class SBFrame;
    friend class SBValue;
    friend class SBTarget;

private:
    // This auto_pointer is made in the constructor and is always valid.
    mutable std::unique_ptr<lldb_private::EvaluateExpressionOptions> m_opaque_ap;
};

} // namespace lldb

#endif  // LLDB_SBExpressionOptions_h_
