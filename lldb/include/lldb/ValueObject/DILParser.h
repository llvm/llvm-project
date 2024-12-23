//===-- DILParser.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_VALUEOBJECT_DILPARSER_H_
#define LLDB_VALUEOBJECT_DILPARSER_H_

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "lldb/Target/ExecutionContextScope.h"
#include "lldb/Utility/Status.h"
#include "lldb/ValueObject/DILAST.h"
#include "lldb/ValueObject/DILLexer.h"

namespace lldb_private {

namespace dil {

enum class ErrorCode : unsigned char {
  kOk = 0,
  kInvalidExpressionSyntax,
  kUndeclaredIdentifier,
  kUnknown,
};

std::string FormatDiagnostics(std::shared_ptr<std::string> input_expr,
                              const std::string &message, uint32_t loc);

/// Pure recursive descent parser for C++ like expressions.
/// EBNF grammar for the parser is described in lldb/docs/dil-expr-lang.ebnf
class DILParser {
public:
  explicit DILParser(std::shared_ptr<std::string> dil_input_expr,
                     std::shared_ptr<ExecutionContextScope> exe_ctx_scope,
                     lldb::DynamicValueType use_dynamic, bool use_synthetic,
                     bool fragile_ivar, bool check_ptr_vs_member);

  DILASTNodeUP Run(Status &error);

  ~DILParser() { m_ctx_scope.reset(); }

  bool UseSynthetic() { return m_use_synthetic; }

  lldb::DynamicValueType UseDynamic() { return m_use_dynamic; }

  using PtrOperator = std::tuple<dil::TokenKind, uint32_t>;

private:
  DILASTNodeUP ParseExpression();
  DILASTNodeUP ParsePrimaryExpression();

  std::string ParseNestedNameSpecifier();

  std::string ParseIdExpression();
  std::string ParseUnqualifiedId();

  void ConsumeToken();

  void BailOut(ErrorCode error_code, const std::string &error, uint32_t loc);

  void BailOut(Status error);

  void Expect(dil::TokenKind kind);

  std::string TokenDescription(const DILToken &token);

  template <typename... Ts> void ExpectOneOf(dil::TokenKind k, Ts... ks);

  void TentativeParsingRollback(uint32_t saved_idx) {
    m_error.Clear();
    m_dil_lexer.ResetTokenIdx(saved_idx);
    m_dil_token = m_dil_lexer.GetCurrentToken();
  }

  // Parser doesn't own the evaluation context. The produced AST may depend on
  // it (for example, for source locations), so it's expected that expression
  // context will outlive the parser.
  std::shared_ptr<ExecutionContextScope> m_ctx_scope;

  std::shared_ptr<std::string> m_input_expr;
  // The token lexer is stopped at (aka "current token").
  DILToken m_dil_token;
  // Holds an error if it occures during parsing.
  Status m_error;

  lldb::DynamicValueType m_use_dynamic;
  bool m_use_synthetic;
  bool m_fragile_ivar;
  bool m_check_ptr_vs_member;
  DILLexer m_dil_lexer;
}; // class DILParser

} // namespace dil

} // namespace lldb_private

#endif // LLDB_VALUEOBJECT_DILPARSER_H_
