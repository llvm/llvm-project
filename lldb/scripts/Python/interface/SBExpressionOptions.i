//===-- SWIG interface for SBExpressionOptions -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"A container for options to use when evaluating expressions."
) SBExpressionOptions;

class SBExpressionOptions
{
friend class SBFrame;
friend class SBValue;

public:
    SBExpressionOptions();

    SBExpressionOptions (const lldb::SBExpressionOptions &rhs);
    
    ~SBExpressionOptions();

    bool
    GetCoerceResultToId () const;
    
    %feature("docstring", "Sets whether to coerce the expression result to ObjC id type after evaluation.") SetCoerceResultToId;
    
    void
    SetCoerceResultToId (bool coerce = true);
    
    bool
    GetUnwindOnError () const;
    
    %feature("docstring", "Sets whether to unwind the expression stack on error.") SetUnwindOnError;
    
    void
    SetUnwindOnError (bool unwind = true);
    
    bool
    GetIgnoreBreakpoints () const;
    
    %feature("docstring", "Sets whether to ignore breakpoint hits while running expressions.") SetUnwindOnError;
    
    void
    SetIgnoreBreakpoints (bool ignore = true);
    
    lldb::DynamicValueType
    GetFetchDynamicValue () const;
    
    %feature("docstring", "Sets whether to cast the expression result to its dynamic type.") SetFetchDynamicValue;
    
    void
    SetFetchDynamicValue (lldb::DynamicValueType dynamic = lldb::eDynamicCanRunTarget);

    uint32_t
    GetTimeoutInMicroSeconds () const;
    
    %feature("docstring", "Sets the timeout in microseconds to run the expression for. If try all threads is set to true and the expression doesn't complete within the specified timeout, all threads will be resumed for the same timeout to see if the expresson will finish.") SetTimeoutInMicroSeconds;
    void
    SetTimeoutInMicroSeconds (uint32_t timeout = 0);
    
    bool
    GetTryAllThreads () const;
    
    %feature("docstring", "Sets whether to run all threads if the expression does not complete on one thread.") SetTryAllThreads;
    void
    SetTryAllThreads (bool run_others = true);
    
protected:

    SBExpressionOptions (lldb_private::EvaluateExpressionOptions &expression_options);

    lldb_private::EvaluateExpressionOptions *
    get () const;

    lldb_private::EvaluateExpressionOptions &
    ref () const;

private:
    // This auto_pointer is made in the constructor and is always valid.
    mutable std::unique_ptr<lldb_private::EvaluateExpressionOptions> m_opaque_ap;
};

} // namespace lldb
