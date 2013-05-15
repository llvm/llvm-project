//===-- ClangExpressionParser.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ClangExpressionParser_h_
#define liblldb_ClangExpressionParser_h_

#include "lldb/lldb-public.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/ClangForward.h"
#include "lldb/Core/Error.h"
#include "lldb/Expression/IRForTarget.h"

#include <string>
#include <vector>

namespace lldb_private
{

class IRExecutionUnit;
    
//----------------------------------------------------------------------
/// @class ClangExpressionParser ClangExpressionParser.h "lldb/Expression/ClangExpressionParser.h"
/// @brief Encapsulates an instance of Clang that can parse expressions.
///
/// ClangExpressionParser is responsible for preparing an instance of
/// ClangExpression for execution.  ClangExpressionParser uses ClangExpression
/// as a glorified parameter list, performing the required parsing and
/// conversion to formats (DWARF bytecode, or JIT compiled machine code)
/// that can be executed.
//----------------------------------------------------------------------
class ClangExpressionParser
{
public:
    //------------------------------------------------------------------
    /// Constructor
    ///
    /// Initializes class variabes.
    ///
    /// @param[in] exe_scope,
    ///     If non-NULL, an execution context scope that can help to 
    ///     correctly create an expression with a valid process for 
    ///     optional tuning Objective-C runtime support. Can be NULL.
    ///
    /// @param[in] expr
    ///     The expression to be parsed.
    //------------------------------------------------------------------
    ClangExpressionParser (ExecutionContextScope *exe_scope,
                           ClangExpression &expr);
    
    //------------------------------------------------------------------
    /// Destructor
    //------------------------------------------------------------------
    ~ClangExpressionParser ();
    
    //------------------------------------------------------------------
    /// Parse a single expression and convert it to IR using Clang.  Don't
    /// wrap the expression in anything at all.
    ///
    /// @param[in] stream
    ///     The stream to print errors to.
    ///
    /// @return
    ///     The number of errors encountered during parsing.  0 means
    ///     success.
    //------------------------------------------------------------------
    unsigned
    Parse (Stream &stream);
    
    //------------------------------------------------------------------
    /// Ready an already-parsed expression for execution, possibly
    /// evaluating it statically.
    ///
    /// @param[out] func_addr
    ///     The address to which the function has been written.
    ///
    /// @param[out] func_end
    ///     The end of the function's allocated memory region.  (func_addr
    ///     and func_end do not delimit an allocated region; the allocated
    ///     region may begin before func_addr.)
    ///
    /// @param[in] execution_unit_ap
    ///     After parsing, ownership of the execution unit for
    ///     for the expression is handed to this unique pointer.
    ///
    /// @param[in] exe_ctx
    ///     The execution context to write the function into.
    ///
    /// @param[out] evaluated_statically
    ///     Set to true if the expression could be interpreted statically;
    ///     untouched otherwise.
    ///
    /// @param[out] const_result
    ///     If the result of the expression is constant, and the
    ///     expression has no side effects, this is set to the result of the 
    ///     expression.
    ///
    /// @param[in] execution_policy
    ///     Determines whether the expression must be JIT-compiled, must be
    ///     evaluated statically, or whether this decision may be made
    ///     opportunistically.
    ///
    /// @return
    ///     An error code indicating the success or failure of the operation.
    ///     Test with Success().
    //------------------------------------------------------------------
    Error
    PrepareForExecution (lldb::addr_t &func_addr,
                         lldb::addr_t &func_end,
                         std::unique_ptr<IRExecutionUnit> &execution_unit_ap,
                         ExecutionContext &exe_ctx,
                         bool &can_interpret,
                         lldb_private::ExecutionPolicy execution_policy);
        
    //------------------------------------------------------------------
    /// Disassemble the machine code for a JITted function from the target 
    /// process's memory and print the result to a stream.
    ///
    /// @param[in] stream
    ///     The stream to print disassembly to.
    ///
    /// @param[in] exc_context
    ///     The execution context to get the machine code from.
    ///
    /// @return
    ///     The error generated.  If .Success() is true, disassembly succeeded.
    //------------------------------------------------------------------
    Error
    DisassembleFunction (Stream &stream, 
                         ExecutionContext &exe_ctx);
    
private:
    ClangExpression &                       m_expr;                 ///< The expression to be parsed
    std::unique_ptr<llvm::LLVMContext>       m_llvm_context;         ///< The LLVM context to generate IR into
    std::unique_ptr<clang::FileManager>      m_file_manager;         ///< The Clang file manager object used by the compiler
    std::unique_ptr<clang::CompilerInstance> m_compiler;             ///< The Clang compiler used to parse expressions into IR
    std::unique_ptr<clang::Builtin::Context> m_builtin_context;      ///< Context for Clang built-ins
    std::unique_ptr<clang::SelectorTable>    m_selector_table;       ///< Selector table for Objective-C methods
    std::unique_ptr<clang::ASTContext>       m_ast_context;          ///< The AST context used to hold types and names for the parser
    std::unique_ptr<clang::CodeGenerator>    m_code_generator;       ///< The Clang object that generates IR
    std::unique_ptr<IRExecutionUnit>         m_execution_unit;       ///< The container for the finished Module
};
    
}

#endif  // liblldb_ClangExpressionParser_h_
