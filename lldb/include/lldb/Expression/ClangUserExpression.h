//===-- ClangUserExpression.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ClangUserExpression_h_
#define liblldb_ClangUserExpression_h_

// C Includes
// C++ Includes
#include <string>
#include <map>
#include <vector>

// Other libraries and framework includes
// Project includes

#include "lldb/lldb-forward.h"
#include "lldb/lldb-private.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/ClangForward.h"
#include "lldb/Expression/ClangExpression.h"
#include "lldb/Expression/ClangExpressionVariable.h"
#include "lldb/Expression/IRForTarget.h"
#include "lldb/Expression/Materializer.h"
#include "lldb/Symbol/TaggedASTType.h"
#include "lldb/Target/ExecutionContext.h"

#include "llvm/ExecutionEngine/JITMemoryManager.h"

namespace lldb_private 
{

//----------------------------------------------------------------------
/// @class ClangUserExpression ClangUserExpression.h "lldb/Expression/ClangUserExpression.h"
/// @brief Encapsulates a single expression for use with Clang
///
/// LLDB uses expressions for various purposes, notably to call functions
/// and as a backend for the expr command.  ClangUserExpression encapsulates
/// the objects needed to parse and interpret or JIT an expression.  It
/// uses the Clang parser to produce LLVM IR from the expression.
//----------------------------------------------------------------------
class ClangUserExpression : public ClangExpression
{
public:
    typedef std::shared_ptr<ClangUserExpression> ClangUserExpressionSP;
    
    enum { kDefaultTimeout = 500000u };
    //------------------------------------------------------------------
    /// Constructor
    ///
    /// @param[in] expr
    ///     The expression to parse.
    ///
    /// @param[in] expr_prefix
    ///     If non-NULL, a C string containing translation-unit level
    ///     definitions to be included when the expression is parsed.
    ///
    /// @param[in] language
    ///     If not eLanguageTypeUnknown, a language to use when parsing
    ///     the expression.  Currently restricted to those languages 
    ///     supported by Clang.
    ///
    /// @param[in] desired_type
    ///     If not eResultTypeAny, the type to use for the expression
    ///     result.
    //------------------------------------------------------------------
    ClangUserExpression (const char *expr,
                         const char *expr_prefix,
                         lldb::LanguageType language,
                         ResultType desired_type);
    
    //------------------------------------------------------------------
    /// Destructor
    //------------------------------------------------------------------
    virtual 
    ~ClangUserExpression ();
    
    //------------------------------------------------------------------
    /// Parse the expression
    ///
    /// @param[in] error_stream
    ///     A stream to print parse errors and warnings to.
    ///
    /// @param[in] exe_ctx
    ///     The execution context to use when looking up entities that
    ///     are needed for parsing (locations of functions, types of
    ///     variables, persistent variables, etc.)
    ///
    /// @param[in] execution_policy
    ///     Determines whether interpretation is possible or mandatory.
    ///
    /// @param[in] keep_result_in_memory
    ///     True if the resulting persistent variable should reside in 
    ///     target memory, if applicable.
    ///
    /// @return
    ///     True on success (no errors); false otherwise.
    //------------------------------------------------------------------
    bool
    Parse (Stream &error_stream, 
           ExecutionContext &exe_ctx,
           lldb_private::ExecutionPolicy execution_policy,
           bool keep_result_in_memory);
    
    bool
    CanInterpret ()
    {
        return m_can_interpret;
    }
    
    bool
    MatchesContext (ExecutionContext &exe_ctx);
    
    //------------------------------------------------------------------
    /// Execute the parsed expression
    ///
    /// @param[in] error_stream
    ///     A stream to print errors to.
    ///
    /// @param[in] exe_ctx
    ///     The execution context to use when looking up entities that
    ///     are needed for parsing (locations of variables, etc.)
    ///
    /// @param[in] unwind_on_error
    ///     If true, and the execution stops before completion, we unwind the
    ///     function call, and return the program state to what it was before the
    ///     execution.  If false, we leave the program in the stopped state.
    ///
    /// @param[in] ignore_breakpoints
    ///     If true, ignore breakpoints while executing the expression.
    ///
    /// @param[in] shared_ptr_to_me
    ///     This is a shared pointer to this ClangUserExpression.  This is
    ///     needed because Execute can push a thread plan that will hold onto
    ///     the ClangUserExpression for an unbounded period of time.  So you
    ///     need to give the thread plan a reference to this object that can 
    ///     keep it alive.
    /// 
    /// @param[in] result
    ///     A pointer to direct at the persistent variable in which the
    ///     expression's result is stored.
    ///
    /// @param[in] try_all_threads
    ///     If true, then we will try to run all threads if the function doesn't complete on
    ///     one thread.  See timeout_usec for the interaction of this variable and
    ///     the timeout.
    ///
    /// @param[in] timeout_usec
    ///     Timeout value (0 for no timeout). If try_all_threads is true, then we
    ///     will try on one thread for the lesser of .25 sec and half the total timeout.
    ///     then switch to running all threads, otherwise this will be the total timeout.
    ///
    ///
    /// @return
    ///     A Process::Execution results value.
    //------------------------------------------------------------------
    ExecutionResults
    Execute (Stream &error_stream,
             ExecutionContext &exe_ctx,
             bool unwind_on_error,
             bool ignore_breakpoints,
             ClangUserExpressionSP &shared_ptr_to_me,
             lldb::ClangExpressionVariableSP &result,
             bool try_all_threads,
             uint32_t timeout_usec);
             
    ThreadPlan *
    GetThreadPlanToExecuteJITExpression (Stream &error_stream,
                                         ExecutionContext &exe_ctx);
    
    //------------------------------------------------------------------
    /// Apply the side effects of the function to program state.
    ///
    /// @param[in] error_stream
    ///     A stream to print errors to.
    ///
    /// @param[in] exe_ctx
    ///     The execution context to use when looking up entities that
    ///     are needed for parsing (locations of variables, etc.)
    /// 
    /// @param[in] result
    ///     A pointer to direct at the persistent variable in which the
    ///     expression's result is stored.
    ///
    /// @param[in] function_stack_pointer
    ///     A pointer to the base of the function's stack frame.  This
    ///     is used to determine whether the expession result resides in
    ///     memory that will still be valid, or whether it needs to be
    ///     treated as homeless for the purpose of future expressions.
    ///
    /// @return
    ///     A Process::Execution results value.
    //------------------------------------------------------------------
    bool
    FinalizeJITExecution (Stream &error_stream,
                          ExecutionContext &exe_ctx,
                          lldb::ClangExpressionVariableSP &result,
                          lldb::addr_t function_stack_bottom = LLDB_INVALID_ADDRESS,
                          lldb::addr_t function_stack_top = LLDB_INVALID_ADDRESS);
    
    //------------------------------------------------------------------
    /// Return the string that the parser should parse.  Must be a full
    /// translation unit.
    //------------------------------------------------------------------
    const char *
    Text ()
    {
        return m_transformed_text.c_str();
    }
    
    //------------------------------------------------------------------
    /// Return the string that the user typed.
    //------------------------------------------------------------------
    const char *
    GetUserText ()
    {
        return m_expr_text.c_str();
    }
    
    //------------------------------------------------------------------
    /// Return the function name that should be used for executing the
    /// expression.  Text() should contain the definition of this
    /// function.
    //------------------------------------------------------------------
    const char *
    FunctionName ()
    {
        return "$__lldb_expr";
    }
    
    //------------------------------------------------------------------
    /// Return the language that should be used when parsing.  To use
    /// the default, return eLanguageTypeUnknown.
    //------------------------------------------------------------------
    virtual lldb::LanguageType
    Language ()
    {
        return m_language;
    }
    
    //------------------------------------------------------------------
    /// Return the object that the parser should use when resolving external
    /// values.  May be NULL if everything should be self-contained.
    //------------------------------------------------------------------
    ClangExpressionDeclMap *
    DeclMap ()
    {
        return m_expr_decl_map.get();
    }
    
    //------------------------------------------------------------------
    /// Return the object that the parser should allow to access ASTs.
    /// May be NULL if the ASTs do not need to be transformed.
    ///
    /// @param[in] passthrough
    ///     The ASTConsumer that the returned transformer should send
    ///     the ASTs to after transformation.
    //------------------------------------------------------------------
    clang::ASTConsumer *
    ASTTransformer (clang::ASTConsumer *passthrough);
    
    //------------------------------------------------------------------
    /// Return the desired result type of the function, or 
    /// eResultTypeAny if indifferent.
    //------------------------------------------------------------------
    virtual ResultType
    DesiredResultType ()
    {
        return m_desired_type;
    }
    
    //------------------------------------------------------------------
    /// Return true if validation code should be inserted into the
    /// expression.
    //------------------------------------------------------------------
    bool
    NeedsValidation ()
    {
        return true;
    }
    
    //------------------------------------------------------------------
    /// Return true if external variables in the expression should be
    /// resolved.
    //------------------------------------------------------------------
    bool
    NeedsVariableResolution ()
    {
        return true;
    }

    //------------------------------------------------------------------
    /// Evaluate one expression and return its result.
    ///
    /// @param[in] exe_ctx
    ///     The execution context to use when evaluating the expression.
    ///
    /// @param[in] execution_policy
    ///     Determines whether or not to try using the IR interpreter to
    ///     avoid running the expression on the parser.
    ///
    /// @param[in] language
    ///     If not eLanguageTypeUnknown, a language to use when parsing
    ///     the expression.  Currently restricted to those languages 
    ///     supported by Clang.
    ///
    /// @param[in] unwind_on_error
    ///     True if the thread's state should be restored in the case 
    ///     of an error.
    ///
    /// @param[in] ignore_breakpoints
    ///     If true, ignore breakpoints while executing the expression.
    ///
    /// @param[in] result_type
    ///     If not eResultTypeAny, the type of the desired result.  Will
    ///     result in parse errors if impossible.
    ///
    /// @param[in] expr_cstr
    ///     A C string containing the expression to be evaluated.
    ///
    /// @param[in] expr_prefix
    ///     If non-NULL, a C string containing translation-unit level
    ///     definitions to be included when the expression is parsed.
    ///
    /// @param[in/out] result_valobj_sp
    ///      If execution is successful, the result valobj is placed here.
    ///
    /// @param[in] try_all_threads
    ///     If true, then we will try to run all threads if the function doesn't complete on
    ///     one thread.  See timeout_usec for the interaction of this variable and
    ///     the timeout.
    ///
    /// @param[in] timeout_usec
    ///     Timeout value (0 for no timeout). If try_all_threads is true, then we
    ///     will try on one thread for the lesser of .25 sec and half the total timeout.
    ///     then switch to running all threads, otherwise this will be the total timeout.
    ///
    /// @result
    ///      A Process::ExecutionResults value.  eExecutionCompleted for success.
    //------------------------------------------------------------------
    static ExecutionResults
    Evaluate (ExecutionContext &exe_ctx,
              lldb_private::ExecutionPolicy execution_policy,
              lldb::LanguageType language,
              ResultType desired_type,
              bool unwind_on_error,
              bool ignore_breakpoints,
              const char *expr_cstr,
              const char *expr_prefix,
              lldb::ValueObjectSP &result_valobj_sp,
              bool try_all_threads,
              uint32_t timeout_usec);
              
    static ExecutionResults
    EvaluateWithError (ExecutionContext &exe_ctx,
                       lldb_private::ExecutionPolicy execution_policy,
                       lldb::LanguageType language,
                       ResultType desired_type,
                       bool unwind_on_error,
                       bool ignore_breakpoints,
                       const char *expr_cstr,
                       const char *expr_prefix,
                       lldb::ValueObjectSP &result_valobj_sp,
                       Error &error,
                       bool try_all_threads,
                       uint32_t timeout_usec);
    
    static const Error::ValueType kNoResult = 0x1001; ///< ValueObject::GetError() returns this if there is no result from the expression.
private:
    //------------------------------------------------------------------
    /// Populate m_cplusplus and m_objetivec based on the environment.
    //------------------------------------------------------------------
    
    void
    ScanContext (ExecutionContext &exe_ctx, 
                 lldb_private::Error &err);

    bool
    PrepareToExecuteJITExpression (Stream &error_stream,
                                   ExecutionContext &exe_ctx,
                                   lldb::addr_t &struct_address,
                                   lldb::addr_t &object_ptr,
                                   lldb::addr_t &cmd_ptr);
    
    void
    InstallContext (ExecutionContext &exe_ctx);
    
    bool
    LockAndCheckContext (ExecutionContext &exe_ctx,
                         lldb::TargetSP &target_sp,
                         lldb::ProcessSP &process_sp,
                         lldb::StackFrameSP &frame_sp);
    
    lldb::ProcessWP                             m_process_wp;           ///< The process used as the context for the expression.
    Address                                     m_address;              ///< The address the process is stopped in.
    lldb::addr_t                                m_stack_frame_bottom;   ///< The bottom of the allocated stack frame.
    lldb::addr_t                                m_stack_frame_top;      ///< The top of the allocated stack frame.
    
    std::string                                 m_expr_text;            ///< The text of the expression, as typed by the user
    std::string                                 m_expr_prefix;          ///< The text of the translation-level definitions, as provided by the user
    lldb::LanguageType                          m_language;             ///< The language to use when parsing (eLanguageTypeUnknown means use defaults)
    bool                                        m_allow_cxx;            ///< True if the language allows C++.
    bool                                        m_allow_objc;           ///< True if the language allows Objective-C.
    std::string                                 m_transformed_text;     ///< The text of the expression, as send to the parser
    ResultType                                  m_desired_type;         ///< The type to coerce the expression's result to.  If eResultTypeAny, inferred from the expression.
    
    std::unique_ptr<ClangExpressionDeclMap>      m_expr_decl_map;        ///< The map to use when parsing the expression.
    std::unique_ptr<IRExecutionUnit>             m_execution_unit_ap;    ///< The execution unit the expression is stored in.
    std::unique_ptr<Materializer>                m_materializer_ap;      ///< The materializer to use when running the expression.
    std::unique_ptr<ASTResultSynthesizer>        m_result_synthesizer;   ///< The result synthesizer, if one is needed.
    
    bool                                        m_enforce_valid_object; ///< True if the expression parser should enforce the presence of a valid class pointer in order to generate the expression as a method.
    bool                                        m_cplusplus;            ///< True if the expression is compiled as a C++ member function (true if it was parsed when exe_ctx was in a C++ method).
    bool                                        m_objectivec;           ///< True if the expression is compiled as an Objective-C method (true if it was parsed when exe_ctx was in an Objective-C method).
    bool                                        m_static_method;        ///< True if the expression is compiled as a static (or class) method (currently true if it was parsed when exe_ctx was in an Objective-C class method).
    bool                                        m_needs_object_ptr;     ///< True if "this" or "self" must be looked up and passed in.  False if the expression doesn't really use them and they can be NULL.
    bool                                        m_const_object;         ///< True if "this" is const.
    Target                                     *m_target;               ///< The target for storing persistent data like types and variables.
    
    bool                                        m_can_interpret;        ///< True if the expression could be evaluated statically; false otherwise.
    lldb::addr_t                                m_materialized_address; ///< The address at which the arguments to the expression have been materialized.
    Materializer::DematerializerSP              m_dematerializer_sp;    ///< The dematerializer.
};
    
} // namespace lldb_private

#endif  // liblldb_ClangUserExpression_h_
