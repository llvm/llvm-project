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
#include "lldb/Target/StackFrame.h"
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

llvm::Expected<lldb::TypeSystemSP>
DILGetTypeSystemFromCU(std::shared_ptr<StackFrame> ctx);

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
/// TypeDeclaration builds information about the literal type definition as
/// type is being parsed. It doesn't perform semantic analysis for non-basic
/// types -- e.g. "char&&&" is a valid type declaration.
/// NOTE: CV qualifiers are ignored.
class TypeDeclaration {
public:
  enum class TypeSpecifier {
    kBool,
    kChar,
    kDouble,
    kFloat,
    kInt,
    kLong,
    kLongDouble,
    kLongLong,
    kShort,
    kUnknown,
    kVoid,
  };

  enum class SignSpecifier {
    kUnknown,
    kSigned,
    kUnsigned,
  };

  bool IsEmpty() const { return !m_is_builtin && !m_is_user_type; }

  lldb::BasicType GetBasicType() const;

public:
  // Indicates user-defined typename (e.g. "MyClass", "MyTmpl<int>").
  std::string m_user_typename;

  // Basic type specifier ("void", "char", "intr", "float", "long long", etc.).
  TypeSpecifier m_type_specifier = TypeSpecifier::kUnknown;

  // Signedness specifier ("signed", "unsigned").
  SignSpecifier m_sign_specifier = SignSpecifier::kUnknown;

  // Does the type declaration includes "int" specifier?
  // This is different than `type_specifier_` and is used to detect "int"
  // duplication for types that can be combined with "int" specifier (e.g.
  // "short int", "long int").
  bool m_has_int_specifier = false;

  // Indicates whether there was an error during parsing.
  bool m_has_error = false;

  // Indicates whether this declaration describes a builtin type.
  bool m_is_builtin = false;

  // Indicates whether this declaration describes a user type.
  bool m_is_user_type = false;
}; // class TypeDeclaration

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
  ASTNodeUP ParseNumericLiteral();
  ASTNodeUP ParseIntegerLiteral();
  ASTNodeUP ParseFloatingPointLiteral();
  ASTNodeUP ParseBooleanLiteral();

  ASTNodeUP ParseCastExpression();
  std::optional<CompilerType> ParseTypeId(bool must_be_type_id = false);
  void ParseTypeSpecifierSeq(TypeDeclaration *type_decl);
  bool ParseTypeSpecifier(TypeDeclaration *type_decl);
  std::string ParseTypeName();
  CompilerType ResolveTypeDeclarators(CompilerType type,
                                      const std::vector<Token> &ptr_operators);
  bool IsSimpleTypeSpecifierKeyword(Token token) const;
  bool HandleSimpleTypeSpecifier(TypeDeclaration *type_decl);

  void BailOut(const std::string &error, uint32_t loc, uint16_t err_len);

  void Expect(Token::Kind kind);

  void ExpectOneOf(std::vector<Token::Kind> kinds_vec);

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

namespace llvm {
template <>
struct format_provider<lldb_private::dil::TypeDeclaration::TypeSpecifier> {
  static void format(const lldb_private::dil::TypeDeclaration::TypeSpecifier &t,
                     raw_ostream &OS, llvm::StringRef Options) {
    switch (t) {
    case lldb_private::dil::TypeDeclaration::TypeSpecifier::kVoid:
      OS << "void";
      break;
    case lldb_private::dil::TypeDeclaration::TypeSpecifier::kBool:
      OS << "bool";
      break;
    case lldb_private::dil::TypeDeclaration::TypeSpecifier::kChar:
      OS << "char";
      break;
    case lldb_private::dil::TypeDeclaration::TypeSpecifier::kInt:
      OS << "int";
      break;
    case lldb_private::dil::TypeDeclaration::TypeSpecifier::kFloat:
      OS << "float";
      break;
    case lldb_private::dil::TypeDeclaration::TypeSpecifier::kShort:
      OS << "short";
      break;
    case lldb_private::dil::TypeDeclaration::TypeSpecifier::kLong:
      OS << "long";
      break;
    case lldb_private::dil::TypeDeclaration::TypeSpecifier::kLongLong:
      OS << "long long";
      break;
    case lldb_private::dil::TypeDeclaration::TypeSpecifier::kDouble:
      OS << "double";
      break;
    case lldb_private::dil::TypeDeclaration::TypeSpecifier::kLongDouble:
      OS << "long double";
      break;
    default:
      OS << "invalid type specifier";
      break;
    }
  }
};

template <>
struct format_provider<lldb_private::dil::TypeDeclaration::SignSpecifier> {
  static void format(const lldb_private::dil::TypeDeclaration::SignSpecifier &t,
                     raw_ostream &OS, llvm::StringRef Options) {
    switch (t) {
    case lldb_private::dil::TypeDeclaration::SignSpecifier::kSigned:
      OS << "signed";
      break;
    case lldb_private::dil::TypeDeclaration::SignSpecifier::kUnsigned:
      OS << "unsigned";
      break;
    default:
      OS << "invalid sign specifier";
      break;
    }
  }
};
} // namespace llvm

#endif // LLDB_VALUEOBJECT_DILPARSER_H
