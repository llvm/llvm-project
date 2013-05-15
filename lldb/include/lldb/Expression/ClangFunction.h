//===-- ClangFunction.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_ClangFunction_h_
#define lldb_ClangFunction_h_

// C Includes
// C++ Includes
#include <vector>
#include <list>
// Other libraries and framework includes
// Project includes
#include "lldb/Core/ClangForward.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/ValueObjectList.h"
#include "lldb/Expression/ClangExpression.h"
#include "lldb/Target/Process.h"

namespace lldb_private
{
    
class ASTStructExtractor;
class ClangExpressionParser;

//----------------------------------------------------------------------
/// @class ClangFunction ClangFunction.h "lldb/Expression/ClangFunction.h"
/// @brief Encapsulates a function that can be called.
///
/// A given ClangFunction object can handle a single function signature.
/// Once constructed, it can set up any number of concurrent calls to
/// functions with that signature.
///
/// It performs the call by synthesizing a structure that contains the pointer
/// to the function and the arguments that should be passed to that function,
/// and producing a special-purpose JIT-compiled function that accepts a void*
/// pointing to this struct as its only argument and calls the function in the 
/// struct with the written arguments.  This method lets Clang handle the
/// vagaries of function calling conventions.
///
/// The simplest use of the ClangFunction is to construct it with a
/// function representative of the signature you want to use, then call
/// ExecuteFunction(ExecutionContext &, Stream &, Value &).
///
/// If you need to reuse the arguments for several calls, you can call
/// InsertFunction() followed by WriteFunctionArguments(), which will return
/// the location of the args struct for the wrapper function in args_addr_ref.
///
/// If you need to call the function on the thread plan stack, you can also 
/// call InsertFunction() followed by GetThreadPlanToCallFunction().
///
/// Any of the methods that take arg_addr_ptr or arg_addr_ref can be passed
/// a pointer set to LLDB_INVALID_ADDRESS and new structure will be allocated
/// and its address returned in that variable.
/// 
/// Any of the methods that take arg_addr_ptr can be passed NULL, and the
/// argument space will be managed for you.
//----------------------------------------------------------------------    
class ClangFunction : public ClangExpression
{
    friend class ASTStructExtractor;
public:
	//------------------------------------------------------------------
	/// Constructor
    ///
    /// @param[in] exe_scope
    ///     An execution context scope that gets us at least a target and 
    ///     process.
    ///
    /// @param[in] function_ptr
    ///     The default function to be called.  Can be overridden using
    ///     WriteFunctionArguments().
    ///
    /// @param[in] ast_context
    ///     The AST context to evaluate argument types in.
    ///
    /// @param[in] arg_value_list
    ///     The default values to use when calling this function.  Can
    ///     be overridden using WriteFunctionArguments().
	//------------------------------------------------------------------  
	ClangFunction (ExecutionContextScope &exe_scope,
                   Function &function_ptr, 
                   ClangASTContext *ast_context, 
                   const ValueList &arg_value_list);
    
    //------------------------------------------------------------------
	/// Constructor
    ///
    /// @param[in] exe_scope
    ///     An execution context scope that gets us at least a target and 
    ///     process.
    ///
    /// @param[in] ast_context
    ///     The AST context to evaluate argument types in.
    ///
    /// @param[in] return_qualtype
    ///     An opaque Clang QualType for the function result.  Should be
    ///     defined in ast_context.
    ///
    /// @param[in] function_address
    ///     The address of the function to call.
    ///
    /// @param[in] arg_value_list
    ///     The default values to use when calling this function.  Can
    ///     be overridden using WriteFunctionArguments().
	//------------------------------------------------------------------
	ClangFunction (ExecutionContextScope &exe_scope,
                   ClangASTContext *ast_context, 
                   void *return_qualtype, 
                   const Address& function_address, 
                   const ValueList &arg_value_list);
    
    //------------------------------------------------------------------
	/// Destructor
	//------------------------------------------------------------------
	virtual 
    ~ClangFunction();

    //------------------------------------------------------------------
	/// Compile the wrapper function
    ///
    /// @param[in] errors
    ///     The stream to print parser errors to.
    ///
    /// @return
    ///     The number of errors.
	//------------------------------------------------------------------
    unsigned
    CompileFunction (Stream &errors);
    
    //------------------------------------------------------------------
	/// Insert the default function wrapper and its default argument struct  
    ///
    /// @param[in] exe_ctx
    ///     The execution context to insert the function and its arguments
    ///     into.
    ///
    /// @param[in,out] args_addr_ref
    ///     The address of the structure to write the arguments into.  May
    ///     be LLDB_INVALID_ADDRESS; if it is, a new structure is allocated
    ///     and args_addr_ref is pointed to it.
    ///
    /// @param[in] errors
    ///     The stream to write errors to.
    ///
    /// @return
    ///     True on success; false otherwise.
	//------------------------------------------------------------------
    bool
    InsertFunction (ExecutionContext &exe_ctx,
                    lldb::addr_t &args_addr_ref,
                    Stream &errors);

    //------------------------------------------------------------------
	/// Insert the default function wrapper (using the JIT)
    ///
    /// @param[in] exe_ctx
    ///     The execution context to insert the function and its arguments
    ///     into.
    ///
    /// @param[in] errors
    ///     The stream to write errors to.
    ///
    /// @return
    ///     True on success; false otherwise.
	//------------------------------------------------------------------
    bool WriteFunctionWrapper (ExecutionContext &exe_ctx, 
                               Stream &errors);
    
    //------------------------------------------------------------------
	/// Insert the default function argument struct  
    ///
    /// @param[in] exe_ctx
    ///     The execution context to insert the function and its arguments
    ///     into.
    ///
    /// @param[in,out] args_addr_ref
    ///     The address of the structure to write the arguments into.  May
    ///     be LLDB_INVALID_ADDRESS; if it is, a new structure is allocated
    ///     and args_addr_ref is pointed to it.
    ///
    /// @param[in] errors
    ///     The stream to write errors to.
    ///
    /// @return
    ///     True on success; false otherwise.
	//------------------------------------------------------------------
    bool WriteFunctionArguments (ExecutionContext &exe_ctx, 
                                 lldb::addr_t &args_addr_ref, 
                                 Stream &errors);
    
    //------------------------------------------------------------------
	/// Insert an argument struct with a non-default function address and
    /// non-default argument values
    ///
    /// @param[in] exe_ctx
    ///     The execution context to insert the function and its arguments
    ///     into.
    ///
    /// @param[in,out] args_addr_ref
    ///     The address of the structure to write the arguments into.  May
    ///     be LLDB_INVALID_ADDRESS; if it is, a new structure is allocated
    ///     and args_addr_ref is pointed to it.
    ///
    /// @param[in] function_address
    ///     The address of the function to call.
    ///
    /// @param[in] arg_values
    ///     The values of the function's arguments.
    ///
    /// @param[in] errors
    ///     The stream to write errors to.
    ///
    /// @return
    ///     True on success; false otherwise.
	//------------------------------------------------------------------
    bool WriteFunctionArguments (ExecutionContext &exe_ctx, 
                                 lldb::addr_t &args_addr_ref, 
                                 Address function_address, 
                                 ValueList &arg_values, 
                                 Stream &errors);

    //------------------------------------------------------------------
	/// [Static] Execute a function, passing it a single void* parameter.
    /// ClangFunction uses this to call the wrapper function.
    ///
    /// @param[in] exe_ctx
    ///     The execution context to insert the function and its arguments
    ///     into.
    ///
    /// @param[in] function_address
    ///     The address of the function in the target process.
    ///
    /// @param[in] void_arg
    ///     The value of the void* parameter.
    ///
    /// @param[in] stop_others
    ///     True if other threads should pause during execution.
    ///
    /// @param[in] try_all_threads
    ///     If the timeout expires, true if other threads should run.  If
    ///     the function may try to take locks, this is useful.
    /// 
    /// @param[in] unwind_on_error
    ///     If true, and the execution stops before completion, we unwind the
    ///     function call, and return the program state to what it was before the
    ///     execution.  If false, we leave the program in the stopped state.
    /// 
    /// @param[in] timeout_usec
    ///     Timeout value (0 for no timeout). If try_all_threads is true, then we
    ///     will try on one thread for the lesser of .25 sec and half the total timeout.
    ///     then switch to running all threads, otherwise this will be the total timeout.
    ///
    /// @param[in] errors
    ///     The stream to write errors to.
    ///
    /// @param[in] this_arg
    ///     If non-NULL, the function is invoked like a C++ method, with the
    ///     value pointed to by the pointer as its 'this' argument.
    ///
    /// @return
    ///     Returns one of the ExecutionResults enum indicating function call status.
	//------------------------------------------------------------------
    static ExecutionResults 
    ExecuteFunction (ExecutionContext &exe_ctx, 
                     lldb::addr_t function_address, 
                     lldb::addr_t &void_arg, 
                     bool stop_others, 
                     bool try_all_threads,
                     bool unwind_on_error,
                     bool ignore_breakpoints,
                     uint32_t timeout_usec,
                     Stream &errors,
                     lldb::addr_t* this_arg = 0);
    
    //------------------------------------------------------------------
    /// Run the function this ClangFunction was created with.
    ///
    /// This simple version will run the function stopping other threads
    /// for a fixed timeout period (1000 usec) and if it does not complete,
    /// we halt the process and try with all threads running.
    ///
    /// @param[in] exe_ctx
    ///     The thread & process in which this function will run.
    ///
    /// @param[in] errors
    ///     Errors will be written here if there are any.
    ///
    /// @param[out] results
    ///     The result value will be put here after running the function.
    ///
    /// @return
    ///     Returns one of the ExecutionResults enum indicating function call status.
    //------------------------------------------------------------------
    ExecutionResults 
    ExecuteFunction(ExecutionContext &exe_ctx, 
                     Stream &errors, 
                     Value &results);
    
    //------------------------------------------------------------------
    /// Run the function this ClangFunction was created with.
    ///
    /// This simple version will run the function obeying the stop_others
    /// argument.  There is no timeout.
    ///
    /// @param[in] exe_ctx
    ///     The thread & process in which this function will run.
    ///
    /// @param[in] errors
    ///     Errors will be written here if there are any.
    ///
    /// @param[in] stop_others
    ///     If \b true, run only this thread, if \b false let all threads run.
    ///
    /// @param[out] results
    ///     The result value will be put here after running the function.
    ///
    /// @return
    ///     Returns one of the ExecutionResults enum indicating function call status.
    //------------------------------------------------------------------
    ExecutionResults 
    ExecuteFunction(ExecutionContext &exe_ctx, 
                     Stream &errors, bool stop_others, 
                     Value &results);
    
    //------------------------------------------------------------------
    /// Run the function this ClangFunction was created with.
    ///
    /// This simple version will run the function on one thread.  If \a timeout_usec
    /// is not zero, we time out after that timeout.  If \a try_all_threads is true, then we will
    /// resume with all threads on, otherwise we halt the process, and eExecutionInterrupted will be returned.
    ///
    /// @param[in] exe_ctx
    ///     The thread & process in which this function will run.
    ///
    /// @param[in] errors
    ///     Errors will be written here if there are any.
    ///
    /// @param[in] timeout_usec
    ///     Timeout value (0 for no timeout). If try_all_threads is true, then we
    ///     will try on one thread for the lesser of .25 sec and half the total timeout.
    ///     then switch to running all threads, otherwise this will be the total timeout.
    ///
    /// @param[in] try_all_threads
    ///     If \b true, run only this thread, if \b false let all threads run.
    ///
    /// @param[out] results
    ///     The result value will be put here after running the function.
    ///
    /// @return
    ///     Returns one of the ExecutionResults enum indicating function call status.
    //------------------------------------------------------------------
    ExecutionResults 
    ExecuteFunction(ExecutionContext &exe_ctx, 
                    Stream &errors, 
                    uint32_t single_thread_timeout_usec, 
                    bool try_all_threads, 
                    Value &results);
    
    //------------------------------------------------------------------
    /// Run the function this ClangFunction was created with.
    ///
    /// This is the full version.
    ///
    /// @param[in] exe_ctx
    ///     The thread & process in which this function will run.
    ///
    /// @param[in] args_addr_ptr
    ///     If NULL, the function will take care of allocating & deallocating the wrapper
    ///     args structure.  Otherwise, if set to LLDB_INVALID_ADDRESS, a new structure
    ///     will be allocated, filled and the address returned to you.  You are responsible
    ///     for deallocating it.  And if passed in with a value other than LLDB_INVALID_ADDRESS,
    ///     this should point to an already allocated structure with the values already written.
    ///
    /// @param[in] errors
    ///     Errors will be written here if there are any.
    ///
    /// @param[in] stop_others
    ///     If \b true, run only this thread, if \b false let all threads run.
    ///
    /// @param[in] timeout_usec
    ///     Timeout value (0 for no timeout). If try_all_threads is true, then we
    ///     will try on one thread for the lesser of .25 sec and half the total timeout.
    ///     then switch to running all threads, otherwise this will be the total timeout.
    ///
    ///
    /// @param[in] try_all_threads
    ///     If \b true, run only this thread, if \b false let all threads run.
    ///
    /// @param[out] results
    ///     The result value will be put here after running the function.
    ///
    /// @return
    ///     Returns one of the ExecutionResults enum indicating function call status.
    //------------------------------------------------------------------
    ExecutionResults 
    ExecuteFunction(ExecutionContext &exe_ctx, 
                    lldb::addr_t *args_addr_ptr, 
                    Stream &errors, 
                    bool stop_others, 
                    uint32_t timeout_usec,
                    bool try_all_threads,
                    bool unwind_on_error,
                    bool ignore_breakpoints,
                    Value &results);
    
    //------------------------------------------------------------------
    /// [static] Get a thread plan to run a function.
    ///
    /// @param[in] exe_ctx
    ///     The execution context to insert the function and its arguments
    ///     into.
    ///
    /// @param[in] func_addr
    ///     The address of the function in the target process.
    ///
    /// @param[in] args_addr_ref
    ///     The value of the void* parameter.
    ///
    /// @param[in] errors
    ///     The stream to write errors to.
    ///
    /// @param[in] stop_others
    ///     True if other threads should pause during execution.
    ///
    /// @param[in] unwind_on_error
    ///     True if the thread plan may simply be discarded if an error occurs.
    ///
    /// @param[in] ignore_breakpoints
    ///     True if the expression execution will ignore breakpoint hits and continue executing.
    ///
    /// @param[in] this_arg
    ///     If non-NULL (and cmd_arg is NULL), the function is invoked like a C++ 
    ///     method, with the value pointed to by the pointer as its 'this' 
    ///     argument.
    ///
    /// @param[in] cmd_arg
    ///     If non-NULL, the function is invoked like an Objective-C method, with
    ///     this_arg in the 'self' slot and cmd_arg in the '_cmd' slot
    ///
    /// @return
    ///     A ThreadPlan for executing the function.
	//------------------------------------------------------------------
    static ThreadPlan *
    GetThreadPlanToCallFunction (ExecutionContext &exe_ctx, 
                                 lldb::addr_t func_addr, 
                                 lldb::addr_t &args_addr_ref, 
                                 Stream &errors, 
                                 bool stop_others, 
                                 bool unwind_on_error,
                                 bool ignore_breakpoints,
                                 lldb::addr_t *this_arg = 0,
                                 lldb::addr_t *cmd_arg = 0);
    
    //------------------------------------------------------------------
    /// Get a thread plan to run the function this ClangFunction was created with.
    ///
    /// @param[in] exe_ctx
    ///     The execution context to insert the function and its arguments
    ///     into.
    ///
    /// @param[in] func_addr
    ///     The address of the function in the target process.
    ///
    /// @param[in] args_addr_ref
    ///     The value of the void* parameter.
    ///
    /// @param[in] errors
    ///     The stream to write errors to.
    ///
    /// @param[in] stop_others
    ///     True if other threads should pause during execution.
    ///
    /// @param[in] unwind_on_error
    ///     True if the thread plan may simply be discarded if an error occurs.
    ///
    /// @return
    ///     A ThreadPlan for executing the function.
	//------------------------------------------------------------------
    ThreadPlan *
    GetThreadPlanToCallFunction (ExecutionContext &exe_ctx, 
                                 lldb::addr_t &args_addr_ref, 
                                 Stream &errors, 
                                 bool stop_others, 
                                 bool unwind_on_error = true,
                                 bool ignore_breakpoints = true)
    {
        return ClangFunction::GetThreadPlanToCallFunction (exe_ctx, 
                                                           m_jit_start_addr, 
                                                           args_addr_ref, 
                                                           errors, 
                                                           stop_others, 
                                                           unwind_on_error,
                                                           ignore_breakpoints);
    }
    
    //------------------------------------------------------------------
    /// Get the result of the function from its struct
    ///
    /// @param[in] exe_ctx
    ///     The execution context to retrieve the result from.
    ///
    /// @param[in] args_addr
    ///     The address of the argument struct.
    ///
    /// @param[in] ret_value
    ///     The value returned by the function.
    ///
    /// @return
    ///     True on success; false otherwise.
	//------------------------------------------------------------------
    bool FetchFunctionResults (ExecutionContext &exe_ctx, 
                               lldb::addr_t args_addr, 
                               Value &ret_value);
    
    //------------------------------------------------------------------
    /// Deallocate the arguments structure
    ///
    /// @param[in] exe_ctx
    ///     The execution context to insert the function and its arguments
    ///     into.
    ///
    /// @param[in] args_addr
    ///     The address of the argument struct.
	//------------------------------------------------------------------
    void DeallocateFunctionResults (ExecutionContext &exe_ctx, 
                                    lldb::addr_t args_addr);
    
    //------------------------------------------------------------------
    /// Interface for ClangExpression
    //------------------------------------------------------------------
    
    //------------------------------------------------------------------
    /// Return the string that the parser should parse.  Must be a full
    /// translation unit.
    //------------------------------------------------------------------
    const char *
    Text ()
    {
        return m_wrapper_function_text.c_str();
    }
    
    //------------------------------------------------------------------
    /// Return the function name that should be used for executing the
    /// expression.  Text() should contain the definition of this
    /// function.
    //------------------------------------------------------------------
    const char *
    FunctionName ()
    {
        return m_wrapper_function_name.c_str();
    }
    
    //------------------------------------------------------------------
    /// Return the object that the parser should use when resolving external
    /// values.  May be NULL if everything should be self-contained.
    //------------------------------------------------------------------
    ClangExpressionDeclMap *
    DeclMap ()
    {
        return NULL;
    }
    
    //------------------------------------------------------------------
    /// Return the object that the parser should use when registering
    /// local variables.  May be NULL if the Expression doesn't care.
    //------------------------------------------------------------------
    ClangExpressionVariableList *
    LocalVariables ()
    {
        return NULL;
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
    /// Return true if validation code should be inserted into the
    /// expression.
    //------------------------------------------------------------------
    bool
    NeedsValidation ()
    {
        return false;
    }
    
    //------------------------------------------------------------------
    /// Return true if external variables in the expression should be
    /// resolved.
    //------------------------------------------------------------------
    bool
    NeedsVariableResolution ()
    {
        return false;
    }
    
    ValueList
    GetArgumentValues () const
    {
        return m_arg_values;
    }
private:
	//------------------------------------------------------------------
	// For ClangFunction only
	//------------------------------------------------------------------

    std::unique_ptr<ClangExpressionParser> m_parser;                 ///< The parser responsible for compiling the function.
    std::unique_ptr<IRExecutionUnit> m_execution_unit_ap;
    
    Function                       *m_function_ptr;                 ///< The function we're going to call.  May be NULL if we don't have debug info for the function.
    Address                         m_function_addr;                ///< If we don't have the FunctionSP, we at least need the address & return type.
    lldb::clang_type_t              m_function_return_qual_type;    ///< The opaque clang qual type for the function return type.
    ClangASTContext                *m_clang_ast_context;            ///< This is the clang_ast_context that we're getting types from the and value, and the function return the function pointer is NULL.

    std::string                     m_wrapper_function_name;        ///< The name of the wrapper function.
    std::string                     m_wrapper_function_text;        ///< The contents of the wrapper function.
    std::string                     m_wrapper_struct_name;          ///< The name of the struct that contains the target function address, arguments, and result.
    std::list<lldb::addr_t>         m_wrapper_args_addrs;           ///< The addresses of the arguments to the wrapper function.
    
    bool                            m_struct_valid;                 ///< True if the ASTStructExtractor has populated the variables below.
    
	//------------------------------------------------------------------
	/// These values are populated by the ASTStructExtractor
    size_t                          m_struct_size;                  ///< The size of the argument struct, in bytes.
    std::vector<uint64_t>           m_member_offsets;               ///< The offset of each member in the struct, in bytes.
    uint64_t                        m_return_size;                  ///< The size of the result variable, in bytes.
    uint64_t                        m_return_offset;                ///< The offset of the result variable in the struct, in bytes.
    //------------------------------------------------------------------

    ValueList                       m_arg_values;                   ///< The default values of the arguments.
    
    bool                            m_compiled;                     ///< True if the wrapper function has already been parsed.
    bool                            m_JITted;                       ///< True if the wrapper function has already been JIT-compiled.
};

} // Namespace lldb_private

#endif  // lldb_ClangFunction_h_
