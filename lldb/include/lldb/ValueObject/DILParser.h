//===-- DILParser.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_DIL_PARSER_H_
#define LLDB_DIL_PARSER_H_

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "clang/Basic/FileManager.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/LiteralSupport.h"
#include "clang/Lex/ModuleLoader.h"
#include "clang/Lex/Preprocessor.h"
#include "lldb/Target/ExecutionContextScope.h"
#include "lldb/Utility/Status.h"
#include "lldb/ValueObject/DILAST.h"

namespace lldb_private {

std::string TypeDescription(CompilerType type);

enum class ErrorCode : unsigned char {
  kOk = 0,
  kInvalidExpressionSyntax,
  kInvalidNumericLiteral,
  kInvalidOperandType,
  kUndeclaredIdentifier,
  kNotImplemented,
  kUBDivisionByZero,
  kUBDivisionByMinusOne,
  kUBInvalidCast,
  kUBInvalidShift,
  kUBNullPtrArithmetic,
  kUBInvalidPtrDiff,
  kSubscriptOutOfRange,
  kUnknown,
};

void SetUbStatus(Status& error, ErrorCode code);

// clang::SourceManager wrapper which takes ownership of the expression string.
class DILSourceManager {
 public:
  static std::shared_ptr<DILSourceManager> Create(std::string expr);

  // This class cannot be safely moved because of the dependency between `m_expr`
  // and `m_smff`. Users are supposed to pass around the shared pointer.
  DILSourceManager(DILSourceManager&&) = delete;
  DILSourceManager(const DILSourceManager&) = delete;
  DILSourceManager& operator=(DILSourceManager const&) = delete;

  clang::SourceManager& GetSourceManager() const { return m_smff->get(); }

 private:
  explicit DILSourceManager(std::string expr);

 private:
  // Store the expression, since SourceManagerForFile doesn't take the
  // ownership.
  std::string m_expr;
  std::unique_ptr<clang::SourceManagerForFile> m_smff;
};

// TypeDeclaration builds information about the literal type definition as type
// is being parsed. It doesn't perform semantic analysis for non-basic types --
// e.g. "char&&&" is a valid type declaration.
// NOTE: CV qualifiers are ignored.
class TypeDeclaration {
 public:
  enum class TypeSpecifier {
    kUnknown,
    kVoid,
    kBool,
    kChar,
    kShort,
    kInt,
    kLong,
    kLongLong,
    kFloat,
    kDouble,
    kLongDouble,
    kWChar,
    kChar16,
    kChar32,
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

  // Basic type specifier ("void", "char", "int", "long", "long long", etc).
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

class BuiltinFunctionDef {
 public:
  BuiltinFunctionDef(std::string name, CompilerType return_type,
                     std::vector<CompilerType> arguments)
      : m_name(std::move(name)),
        m_return_type(std::move(return_type)),
        m_arguments(std::move(arguments)) {}

  std::string m_name;
  CompilerType m_return_type;
  std::vector<CompilerType> m_arguments;
}; // class BuiltinFunctionDef

// Pure recursive descent parser for C++ like expressions.
// EBNF grammar for the parser is described in lldb/docs/dil-expr-lang.ebnf
class DILParser {
 public:
  explicit DILParser(std::shared_ptr<DILSourceManager> dil_sm,
                     std::shared_ptr<ExecutionContextScope> exe_ctx_scope,
                     lldb::DynamicValueType use_dynamic,
                     bool use_synthetic);

  ParseResult Run(Status& error);

  ~DILParser() { m_ctx_scope.reset(); }

  bool UseSynthetic() { return m_use_synthetic; }

  lldb::DynamicValueType UseDynamic() { return m_use_dynamic; }

  using PtrOperator = std::tuple<clang::tok::TokenKind, clang::SourceLocation>;

 private:
  ParseResult ParseExpression();
  ParseResult ParseAssignmentExpression();
  ParseResult ParseLogicalOrExpression();
  ParseResult ParseLogicalAndExpression();
  ParseResult ParseInclusiveOrExpression();
  ParseResult ParseExclusiveOrExpression();
  ParseResult ParseAndExpression();
  ParseResult ParseEqualityExpression();
  ParseResult ParseRelationalExpression();
  ParseResult ParseShiftExpression();
  ParseResult ParseAdditiveExpression();
  ParseResult ParseMultiplicativeExpression();
  ParseResult ParseCastExpression();
  ParseResult ParseUnaryExpression();
  ParseResult ParsePostfixExpression();
  ParseResult ParsePrimaryExpression();

  std::optional<CompilerType> ParseTypeId(bool must_be_type_id = false);
  void ParseTypeSpecifierSeq(TypeDeclaration* type_decl);
  bool ParseTypeSpecifier(TypeDeclaration* type_decl);
  std::string ParseNestedNameSpecifier();
  std::string ParseTypeName();

  std::string ParseTemplateArgumentList();
  std::string ParseTemplateArgument();

  PtrOperator ParsePtrOperator();
  CompilerType ResolveTypeDeclarators(
      CompilerType type,
      const std::vector<PtrOperator>& ptr_operators);

  bool IsSimpleTypeSpecifierKeyword(clang::Token token) const;
  bool IsCvQualifier(clang::Token token) const;
  bool IsPtrOperator(clang::Token token) const;
  bool HandleSimpleTypeSpecifier(TypeDeclaration* type_decl);

  std::string ParseIdExpression();
  std::string ParseUnqualifiedId();
  ParseResult ParseNumericLiteral();
  ParseResult ParseBooleanLiteral();
  ParseResult ParseCharLiteral();
  ParseResult ParseStringLiteral();
  ParseResult ParsePointerLiteral();
  ParseResult ParseNumericConstant(clang::Token token);
  ParseResult ParseFloatingLiteral(clang::NumericLiteralParser& literal,
                                  clang::Token token);
  ParseResult ParseIntegerLiteral(clang::NumericLiteralParser& literal,
                                 clang::Token token);
  ParseResult ParseBuiltinFunction(clang::SourceLocation loc,
                                  std::unique_ptr<BuiltinFunctionDef> func_def);

  bool ImplicitConversionIsAllowed(CompilerType src, CompilerType dst,
                                   bool is_src_literal_zero = false);
  ParseResult InsertImplicitConversion(ParseResult expr, CompilerType type);

  void ConsumeToken();

  void BailOut(ErrorCode error_code, const std::string& error,
               clang::SourceLocation loc);
  void Expect(clang::tok::TokenKind kind);

  std::string TokenDescription(const clang::Token& token);

  std::string FormatDiagnostics(clang::SourceManager& sm,
                                const std::string& message,
                                clang::SourceLocation loc);

  template <typename... Ts>
  void ExpectOneOf(clang::tok::TokenKind k, Ts... ks);

  ParseResult BuildCStyleCast(CompilerType type, ParseResult rhs,
                             clang::SourceLocation location);
  ParseResult BuildCxxCast(clang::tok::TokenKind kind, CompilerType type,
                          ParseResult rhs, clang::SourceLocation location);
  ParseResult BuildCxxDynamicCast(CompilerType type, ParseResult rhs,
                                 clang::SourceLocation location);
  ParseResult BuildCxxStaticCast(CompilerType type, ParseResult rhs,
                                clang::SourceLocation location);
  ParseResult BuildCxxStaticCastToScalar(CompilerType type, ParseResult rhs,
                                        clang::SourceLocation location);
  ParseResult BuildCxxStaticCastToEnum(CompilerType type, ParseResult rhs,
                                      clang::SourceLocation location);
  ParseResult BuildCxxStaticCastToPointer(CompilerType type, ParseResult rhs,
                                         clang::SourceLocation location);
  ParseResult BuildCxxStaticCastToNullPtr(CompilerType type, ParseResult rhs,
                                         clang::SourceLocation location);
  ParseResult BuildCxxStaticCastToReference(CompilerType type, ParseResult rhs,
                                           clang::SourceLocation location);
  ParseResult BuildCxxStaticCastForInheritedTypes(
      CompilerType type, ParseResult rhs, clang::SourceLocation location);
  ParseResult BuildCxxReinterpretCast(CompilerType type, ParseResult rhs,
                                     clang::SourceLocation location);
  ParseResult BuildUnaryOp(UnaryOpKind kind, ParseResult rhs,
                          clang::SourceLocation location);
  ParseResult BuildIncrementDecrement(UnaryOpKind kind, ParseResult rhs,
                                     clang::SourceLocation location);
  ParseResult BuildBinaryOp(BinaryOpKind kind, ParseResult lhs, ParseResult rhs,
                           clang::SourceLocation location);
  CompilerType PrepareBinaryAddition(ParseResult& lhs, ParseResult& rhs,
                                     clang::SourceLocation location,
                                     bool is_comp_assign);
  CompilerType PrepareBinarySubtraction(ParseResult& lhs, ParseResult& rhs,
                                        clang::SourceLocation location,
                                        bool is_comp_assign);
  CompilerType PrepareBinaryMulDiv(ParseResult& lhs, ParseResult& rhs,
                                   bool is_comp_assign);
  CompilerType PrepareBinaryRemainder(ParseResult& lhs, ParseResult& rhs,
                                      bool is_comp_assign);
  CompilerType PrepareBinaryBitwise(ParseResult& lhs, ParseResult& rhs,
                                    bool is_comp_assign);
  CompilerType PrepareBinaryShift(ParseResult& lhs, ParseResult& rhs,
                                  bool is_comp_assign);
  CompilerType PrepareBinaryComparison(BinaryOpKind kind, ParseResult& lhs,
                                 ParseResult& rhs,
                                 clang::SourceLocation location);
  CompilerType PrepareBinaryLogical(const ParseResult& lhs,
                                    const ParseResult& rhs);
  ParseResult BuildBinarySubscript(ParseResult lhs, ParseResult rhs,
                                  clang::SourceLocation location);
  CompilerType PrepareCompositeAssignment(CompilerType comp_assign_type,
                                          const ParseResult& lhs,
                                          clang::SourceLocation location);
  ParseResult BuildTernaryOp(ParseResult cond, ParseResult lhs, ParseResult rhs,
                            clang::SourceLocation location);
  ParseResult BuildMemberOf(ParseResult lhs, std::string member_id,
                            bool is_arrow,
                            clang::SourceLocation location);

  bool AllowSideEffects() const { return m_allow_side_effects; }

  void SetAllowSideEffects (bool allow_side_effects) {
    m_allow_side_effects = allow_side_effects;
  }

  const IdentifierInfo& GetInfo(const IdentifierNode *node) {
    return node->info();
  }

 private:
  friend class TentativeParsingAction;

  // Parser doesn't own the evaluation context. The produced AST may depend on
  // it (for example, for source locations), so it's expected that expression
  // context will outlive the parser.
  std::shared_ptr<ExecutionContextScope> m_ctx_scope;

  std::shared_ptr<clang::SourceManagerForFile> m_smff;

  std::shared_ptr<DILSourceManager> m_sm;
  // The token lexer is stopped at (aka "current token").
  clang::Token m_token;
  // Holds an error if it occures during parsing.
  Status m_error;

  bool m_allow_side_effects;

  std::unique_ptr<clang::TargetInfo> m_ti;
  std::unique_ptr<clang::LangOptions> m_lang_opts;
  std::unique_ptr<clang::HeaderSearch> m_hs;
  std::unique_ptr<clang::TrivialModuleLoader> m_tml;
  std::unique_ptr<clang::Preprocessor> m_pp;
  lldb::DynamicValueType m_use_dynamic;
  bool m_use_synthetic;
}; // class DILParser


// Enables tentative parsing mode, allowing to rollback the parser state. Call
// Commit() or Rollback() to control the parser state. If neither was called,
// the destructor will assert.
class TentativeParsingAction {
 public:
  TentativeParsingAction(DILParser* parser) : m_parser(parser) {
    m_backtrack_token = m_parser->m_token;
    m_parser->m_pp->EnableBacktrackAtThisPos();
    m_enabled = true;
  }

  ~TentativeParsingAction() {
    assert(!m_enabled &&
           "Tentative parsing wasn't finalized. Did you forget to call "
           "Commit() or Rollback()?");
  }

  void Commit() {
    m_parser->m_pp->CommitBacktrackedTokens();
    m_enabled = false;
  }

  void Rollback() {
    m_parser->m_pp->Backtrack();
    m_parser->m_error.Clear();
    m_parser->m_token = m_backtrack_token;
    m_enabled = false;
  }

 private:
  DILParser* m_parser;
  clang::Token m_backtrack_token;
  bool m_enabled;
}; // class TentativeParsingAction

}  // namespace lldb_private

#endif  // LLDB_DIL_PARSER_H_
