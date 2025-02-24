//===-- DILParser.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_VALUEOBJECT_DILPARSER_H
#define LLDB_VALUEOBJECT_DILPARSER_H

#include "lldb/Target/ExecutionContextScope.h"
#include "lldb/Utility/Status.h"
#include "lldb/ValueObject/DILAST.h"
#include "lldb/ValueObject/DILLexer.h"
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace lldb_private::dil {

enum class ErrorCode : unsigned char {
  kOk = 0,
  kInvalidExpressionSyntax,
  kUndeclaredIdentifier,
  kUnknown,
};

std::string FormatDiagnostics(llvm::StringRef input_expr,
                              const std::string &message, uint32_t loc);

/// Pure recursive descent parser for C++ like expressions.
/// EBNF grammar for the parser is described in lldb/docs/dil-expr-lang.ebnf
class DILParser {
public:
  static llvm::Expected<ASTNodeUP>
  Parse(llvm::StringRef dil_input_expr, DILLexer lexer,
        std::shared_ptr<ExecutionContextScope> exe_ctx_scope,
        lldb::DynamicValueType use_dynamic, bool use_synthetic,
        bool fragile_ivar, bool check_ptr_vs_member);

  ~DILParser() { m_ctx_scope.reset(); }

  bool UseSynthetic() { return m_use_synthetic; }

  lldb::DynamicValueType UseDynamic() { return m_use_dynamic; }

  using PtrOperator = std::tuple<Token::Kind, uint32_t>;

private:
  explicit DILParser(llvm::StringRef dil_input_expr, DILLexer lexer,
                     std::shared_ptr<ExecutionContextScope> exe_ctx_scope,
                     lldb::DynamicValueType use_dynamic, bool use_synthetic,
                     bool fragile_ivar, bool check_ptr_vs_member,
                     Status &error);

  llvm::Expected<ASTNodeUP> Run();

  ASTNodeUP ParseExpression();
  ASTNodeUP ParsePrimaryExpression();

  std::string ParseNestedNameSpecifier();

  std::string ParseIdExpression();
  std::string ParseUnqualifiedId();

  void BailOut(ErrorCode error_code, const std::string &error, uint32_t loc);

  void BailOut(Status error);

  void Expect(Token::Kind kind);

  std::string TokenDescription(const Token &token);

  void TentativeParsingRollback(uint32_t saved_idx) {
    m_error.Clear();
    m_dil_lexer.ResetTokenIdx(saved_idx);
  }

  Token CurToken() { return m_dil_lexer.GetCurrentToken(); }

  // Parser doesn't own the evaluation context. The produced AST may depend on
  // it (for example, for source locations), so it's expected that expression
  // context will outlive the parser.
  std::shared_ptr<ExecutionContextScope> m_ctx_scope;

  llvm::StringRef m_input_expr;

  DILLexer m_dil_lexer;

  // Holds an error if it occures during parsing.
  Status &m_error;

  lldb::DynamicValueType m_use_dynamic;
  bool m_use_synthetic;
  bool m_fragile_ivar;
  bool m_check_ptr_vs_member;
}; // class DILParser

} // namespace lldb_private::dil

#endif // LLDB_VALUEOBJECT_DILPARSER_H
