//===-- ExpressionParser.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_EXPRESSION_EXPRESSIONPARSER_H
#define LLDB_EXPRESSION_EXPRESSIONPARSER_H

#include "lldb/Utility/CompletionRequest.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-private-enumerations.h"
#include "lldb/lldb-public.h"

namespace lldb_private {

class IRExecutionUnit;

/// \class ExpressionParser ExpressionParser.h
/// "lldb/Expression/ExpressionParser.h" Encapsulates an instance of a
/// compiler that can parse expressions.
///
/// ExpressionParser is the base class for llvm based Expression parsers.
///
/// ExpressionParser represents language-specific compilers that can parse and
/// compile user-provided expressions for evaluation in the debugger. The most
/// common implementation is ClangExpressionParser, which compiles C, C++, and
/// Objective-C expressions.
///
/// These parsers are used whenever LLDB needs to evaluate user expressions,
/// such as:
/// - Interactive expression evaluation (expr command, p command, po command)
/// - Conditional breakpoints with expression conditions
/// - Watchpoint conditions
/// - Variable formatters and summaries with custom code
///
/// LLDB instantiates expression parsers through language-specific
/// UserExpression subclasses (e.g., ClangUserExpression). The UserExpression
/// creates an ExpressionParser instance during its Parse() method, passing in
/// the expression text and execution context. The parser is then responsible
/// for:
/// 1. Parsing the source code text into an abstract syntax tree (AST)
/// 2. Performing semantic analysis and type checking
/// 3. Generating LLVM IR (intermediate representation) from the AST
/// 4. Optimizing the IR
/// 5. Preparing the IR for execution (JIT compilation or interpretation)
///
/// The parser works in conjunction with several other components:
/// - Expression: The base class representing the expression being parsed
/// - IRExecutionUnit: Manages the JIT-compiled code and execution
/// - DiagnosticManager: Collects errors and warnings during parsing
/// - ExpressionSourceCode: Wraps user expressions in necessary scaffolding
///   code (e.g., wrapping a C++ expression in a function body)
///
/// Subclasses should be careful to:
/// - Properly handle target-specific compilation flags and settings
/// - Set up correct include paths for header files
/// - Handle both interpreted and JIT-compiled execution paths
/// - Manage memory and resources for the compiler instances
/// - Provide accurate diagnostic information for parse errors
/// - Support debug info generation when requested
class ExpressionParser {
public:
  /// Constructor
  ///
  /// Initializes class variables.
  ///
  /// \param[in] exe_scope
  ///     If non-NULL, an execution context scope that can help to
  ///     correctly create an expression with a valid process for
  ///     optional tuning Objective-C runtime support. Can be NULL.
  ///
  /// \param[in] expr
  ///     The expression to be parsed.
  ExpressionParser(ExecutionContextScope *exe_scope, Expression &expr,
                   bool generate_debug_info)
      : m_expr(expr), m_generate_debug_info(generate_debug_info) {}

  /// Destructor
  virtual ~ExpressionParser() = default;

  /// Attempts to find possible command line completions for the given
  /// expression.
  ///
  /// \param[out] request
  ///     The completion request to fill out. The completion should be a string
  ///     that would complete the current token at the cursor position.
  ///     Note that the string in the list replaces the current token
  ///     in the command line.
  ///
  /// \param[in] line
  ///     The line with the completion cursor inside the expression as a string.
  ///     The first line in the expression has the number 0.
  ///
  /// \param[in] pos
  ///     The character position in the line with the completion cursor.
  ///     If the value is 0, then the cursor is on top of the first character
  ///     in the line (i.e. the user has requested completion from the start of
  ///     the expression).
  ///
  /// \param[in] typed_pos
  ///     The cursor position in the line as typed by the user. If the user
  ///     expression has not been transformed in some form (e.g. wrapping it
  ///     in a function body for C languages), then this is equal to the
  ///     'pos' parameter. The semantics of this value are otherwise equal to
  ///     'pos' (e.g. a value of 0 means the cursor is at start of the
  ///     expression).
  ///
  /// \return
  ///     True if we added any completion results to the output;
  ///     false otherwise.
  virtual bool Complete(CompletionRequest &request, unsigned line, unsigned pos,
                        unsigned typed_pos) = 0;

  /// Try to use the FixIts in the diagnostic_manager to rewrite the
  /// expression.  If successful, the rewritten expression is stored in the
  /// diagnostic_manager, get it out with GetFixedExpression.
  ///
  /// \param[in] diagnostic_manager
  ///     The diagnostic manager containing fixit's to apply.
  ///
  /// \return
  ///     \b true if the rewrite was successful, \b false otherwise.
  virtual bool RewriteExpression(DiagnosticManager &diagnostic_manager) {
    return false;
  }

  /// Ready an already-parsed expression for execution, possibly evaluating it
  /// statically.
  ///
  /// \param[out] func_addr
  ///     The address to which the function has been written.
  ///
  /// \param[out] func_end
  ///     The end of the function's allocated memory region.  (func_addr
  ///     and func_end do not delimit an allocated region; the allocated
  ///     region may begin before func_addr.)
  ///
  /// \param[in] execution_unit_sp
  ///     After parsing, ownership of the execution unit for
  ///     for the expression is handed to this shared pointer.
  ///
  /// \param[in] exe_ctx
  ///     The execution context to write the function into.
  ///
  /// \param[out] can_interpret
  ///     Set to true if the expression could be interpreted statically;
  ///     untouched otherwise.
  ///
  /// \param[in] execution_policy
  ///     Determines whether the expression must be JIT-compiled, must be
  ///     evaluated statically, or whether this decision may be made
  ///     opportunistically.
  ///
  /// \return
  ///     An error code indicating the success or failure of the operation.
  ///     Test with Success().
  Status
  PrepareForExecution(lldb::addr_t &func_addr, lldb::addr_t &func_end,
                      std::shared_ptr<IRExecutionUnit> &execution_unit_sp,
                      ExecutionContext &exe_ctx, bool &can_interpret,
                      lldb_private::ExecutionPolicy execution_policy);

  bool GetGenerateDebugInfo() const { return m_generate_debug_info; }

protected:
  virtual Status
  DoPrepareForExecution(lldb::addr_t &func_addr, lldb::addr_t &func_end,
                        std::shared_ptr<IRExecutionUnit> &execution_unit_sp,
                        ExecutionContext &exe_ctx, bool &can_interpret,
                        lldb_private::ExecutionPolicy execution_policy) = 0;

private:
  /// Run all static initializers for an execution unit.
  ///
  /// \param[in] execution_unit_sp
  ///     The execution unit.
  ///
  /// \param[in] exe_ctx
  ///     The execution context to use when running them.  Thread can't be null.
  ///
  /// \return
  ///     The error code indicating the
  Status RunStaticInitializers(lldb::IRExecutionUnitSP &execution_unit_sp,
                               ExecutionContext &exe_ctx);

protected:
  Expression &m_expr; ///< The expression to be parsed
  bool m_generate_debug_info;
};
}

#endif // LLDB_EXPRESSION_EXPRESSIONPARSER_H
