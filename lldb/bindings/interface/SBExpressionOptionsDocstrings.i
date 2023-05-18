%feature("docstring",
"A container for options to use when evaluating expressions."
) lldb::SBExpressionOptions;

%feature("docstring", "Sets whether to coerce the expression result to ObjC id type after evaluation."
) lldb::SBExpressionOptions::SetCoerceResultToId;

%feature("docstring", "Sets whether to unwind the expression stack on error."
) lldb::SBExpressionOptions::SetUnwindOnError;

%feature("docstring", "Sets whether to ignore breakpoint hits while running expressions."
) lldb::SBExpressionOptions::SetIgnoreBreakpoints;

%feature("docstring", "Sets whether to cast the expression result to its dynamic type."
) lldb::SBExpressionOptions::SetFetchDynamicValue;

%feature("docstring", "Sets the timeout in microseconds to run the expression for. If try all threads is set to true and the expression doesn't complete within the specified timeout, all threads will be resumed for the same timeout to see if the expression will finish."
) lldb::SBExpressionOptions::SetTimeoutInMicroSeconds;

%feature("docstring", "Sets the timeout in microseconds to run the expression on one thread before either timing out or trying all threads."
) lldb::SBExpressionOptions::SetOneThreadTimeoutInMicroSeconds;

%feature("docstring", "Sets whether to run all threads if the expression does not complete on one thread."
) lldb::SBExpressionOptions::SetTryAllThreads;

%feature("docstring", "Sets whether to stop other threads at all while running expressions.  If false, TryAllThreads does nothing."
) lldb::SBExpressionOptions::SetStopOthers;

%feature("docstring", "Sets whether to abort expression evaluation if an exception is thrown while executing.  Don't set this to false unless you know the function you are calling traps all exceptions itself."
) lldb::SBExpressionOptions::SetTrapExceptions;

%feature ("docstring", "Sets the language that LLDB should assume the expression is written in"
) lldb::SBExpressionOptions::SetLanguage;

%feature("docstring", "Sets whether to generate debug information for the expression and also controls if a SBModule is generated."
) lldb::SBExpressionOptions::SetGenerateDebugInfo;

%feature("docstring", "Sets whether to produce a persistent result that can be used in future expressions."
) lldb::SBExpressionOptions::SetSuppressPersistentResult;

%feature("docstring", "Gets the prefix to use for this expression."
) lldb::SBExpressionOptions::GetPrefix;

%feature("docstring", "Sets the prefix to use for this expression. This prefix gets inserted after the 'target.expr-prefix' prefix contents, but before the wrapped expression function body."
) lldb::SBExpressionOptions::SetPrefix;

%feature("docstring", "Sets whether to auto-apply fix-it hints to the expression being evaluated."
) lldb::SBExpressionOptions::SetAutoApplyFixIts;

%feature("docstring", "Gets whether to auto-apply fix-it hints to an expression."
) lldb::SBExpressionOptions::GetAutoApplyFixIts;

%feature("docstring", "Sets how often LLDB should retry applying fix-its to an expression."
) lldb::SBExpressionOptions::SetRetriesWithFixIts;

%feature("docstring", "Gets how often LLDB will retry applying fix-its to an expression."
) lldb::SBExpressionOptions::GetRetriesWithFixIts;

%feature("docstring", "Gets whether to JIT an expression if it cannot be interpreted."
) lldb::SBExpressionOptions::GetAllowJIT;

%feature("docstring", "Sets whether to JIT an expression if it cannot be interpreted."
) lldb::SBExpressionOptions::SetAllowJIT;
