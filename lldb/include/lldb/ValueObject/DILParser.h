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
#include "lldb/Utility/DiagnosticsRendering.h"
#include "lldb/Utility/Status.h"
#include "lldb/ValueObject/DILAST.h"
#include "lldb/ValueObject/DILLexer.h"
#include "llvm/Support/Error.h"
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <tuple>
#include <vector>

namespace lldb_private::dil {

enum class ErrorCode : unsigned char {
  kOk = 0,
  kInvalidExpressionSyntax,
  kUndeclaredIdentifier,
  kUnknown,
};

// The following is modeled on class OptionParseError.
class DILDiagnosticError
    : public llvm::ErrorInfo<DILDiagnosticError, DiagnosticError> {
  DiagnosticDetail m_detail;

public:
  using llvm::ErrorInfo<DILDiagnosticError, DiagnosticError>::ErrorInfo;
  DILDiagnosticError(DiagnosticDetail detail)
      : ErrorInfo(make_error_code(std::errc::invalid_argument)),
        m_detail(std::move(detail)) {}

  DILDiagnosticError(llvm::StringRef expr, const std::string &message,
                     uint32_t loc, uint16_t err_len = 1);

  std::unique_ptr<CloneableError> Clone() const override {
    return std::make_unique<DILDiagnosticError>(m_detail);
  }

  llvm::ArrayRef<DiagnosticDetail> GetDetails() const override {
    return m_detail;
  }

  std::string message() const override { return m_detail.rendered; }
};

/// Pure recursive descent parser for C++ like expressions.
/// EBNF grammar for the parser is described in lldb/docs/dil-expr-lang.ebnf
class DILParser {
public:
  static llvm::Expected<ASTNodeUP> Parse(llvm::StringRef dil_input_expr,
                                         DILLexer lexer,
                                         std::shared_ptr<StackFrame> frame_sp,
                                         lldb::DynamicValueType use_dynamic,
                                         bool use_synthetic, bool fragile_ivar,
                                         bool check_ptr_vs_member);

  ~DILParser() = default;

  bool UseSynthetic() { return m_use_synthetic; }

  bool UseFragileIvar() { return m_fragile_ivar; }

  bool CheckPtrVsMember() { return m_check_ptr_vs_member; }

  lldb::DynamicValueType UseDynamic() { return m_use_dynamic; }

private:
  explicit DILParser(llvm::StringRef dil_input_expr, DILLexer lexer,
                     std::shared_ptr<StackFrame> frame_sp,
                     lldb::DynamicValueType use_dynamic, bool use_synthetic,
                     bool fragile_ivar, bool check_ptr_vs_member,
                     llvm::Error &error);

  ASTNodeUP Run();

  ASTNodeUP ParseExpression();
  ASTNodeUP ParseUnaryExpression();
  ASTNodeUP ParsePostfixExpression();
  ASTNodeUP ParsePrimaryExpression();

  std::string ParseNestedNameSpecifier();

  std::string ParseIdExpression();
  std::string ParseUnqualifiedId();
  std::optional<int64_t> ParseIntegerConstant();

  void BailOut(const std::string &error, uint32_t loc, uint16_t err_len);

  void Expect(Token::Kind kind);

  void TentativeParsingRollback(uint32_t saved_idx) {
    if (m_error)
      llvm::consumeError(std::move(m_error));
    m_dil_lexer.ResetTokenIdx(saved_idx);
  }

  Token CurToken() { return m_dil_lexer.GetCurrentToken(); }

  // Parser doesn't own the evaluation context. The produced AST may depend on
  // it (for example, for source locations), so it's expected that expression
  // context will outlive the parser.
  std::shared_ptr<StackFrame> m_ctx_scope;

  llvm::StringRef m_input_expr;

  DILLexer m_dil_lexer;

  // Holds an error if it occures during parsing.
  llvm::Error &m_error;

  lldb::DynamicValueType m_use_dynamic;
  bool m_use_synthetic;
  bool m_fragile_ivar;
  bool m_check_ptr_vs_member;
}; // class DILParser

} // namespace lldb_private::dil

#endif // LLDB_VALUEOBJECT_DILPARSER_H
