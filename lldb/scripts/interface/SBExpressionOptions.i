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
    
    uint32_t
    GetOneThreadTimeoutInMicroSeconds () const;
    
    %feature("docstring", "Sets the timeout in microseconds to run the expression on one thread before either timing out or trying all threads.") SetTimeoutInMicroSeconds;
    void
    SetOneThreadTimeoutInMicroSeconds (uint32_t timeout = 0);
    
    bool
    GetTryAllThreads () const;
    
    %feature("docstring", "Sets whether to run all threads if the expression does not complete on one thread.") SetTryAllThreads;
    void
    SetTryAllThreads (bool run_others = true);
    
    bool
    GetStopOthers () const;
    
    %feature("docstring", "Sets whether to stop other threads at all while running expressins.  If false, TryAllThreads does nothing.") SetTryAllThreads;
    void
    SetStopOthers (bool stop_others = true);
    
    bool
    GetTrapExceptions () const;
    
    %feature("docstring", "Sets whether to abort expression evaluation if an exception is thrown while executing.  Don't set this to false unless you know the function you are calling traps all exceptions itself.") SetTryAllThreads;
    void
    SetTrapExceptions (bool trap_exceptions = true);
    
    %feature ("docstring", "Sets the language that LLDB should assume the expression is written in") SetLanguage;
    void
    SetLanguage (lldb::LanguageType language);

    bool
    GetGenerateDebugInfo ();

    %feature("docstring", "Sets whether to generate debug information for the expression and also controls if a SBModule is generated.") SetGenerateDebugInfo;
    void
    SetGenerateDebugInfo (bool b = true);
    
    bool
    GetSuppressPersistentResult ();
    
    %feature("docstring", "Sets whether to produce a persistent result that can be used in future expressions.") SetSuppressPersistentResult;
    void
    SetSuppressPersistentResult (bool b = false);


    %feature("docstring", "Gets the prefix to use for this expression.") GetPrefix;
    const char *
    GetPrefix () const;

    %feature("docstring", "Sets the prefix to use for this expression. This prefix gets inserted after the 'target.expr-prefix' prefix contents, but before the wrapped expression function body.") SetPrefix;
    void
    SetPrefix (const char *prefix);
    
    %feature("docstring", "Sets whether to auto-apply fix-it hints to the expression being evaluated.") SetAutoApplyFixIts;
    void
    SetAutoApplyFixIts(bool b = true);
    
    %feature("docstring", "Gets whether to auto-apply fix-it hints to an expression.") GetAutoApplyFixIts;
    bool
    GetAutoApplyFixIts();

    bool
    GetTopLevel();

    void
    SetTopLevel(bool b = true);

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
