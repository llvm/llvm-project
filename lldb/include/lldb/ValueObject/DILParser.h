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
#include "lldb/ValueObject/DILLiteralParsers.h"

namespace lldb_private {

namespace dil {

/// Finds the member field with the given name and type, stores the child index
/// corresponding to the field in the idx vector and returns a MemberInfo
/// struct with appropriate information about the field.
std::optional<MemberInfo>
GetFieldWithNameIndexPath(lldb::ValueObjectSP lhs_val_sp,
                          CompilerType type,
                          const std::string &name,
                          std::vector<uint32_t> *idx,
                          CompilerType empty_type,
                          bool use_synthetic, bool is_dynamic, bool is_arrow);

std::tuple<std::optional<MemberInfo>, std::vector<uint32_t>>
GetMemberInfo(lldb::ValueObjectSP lhs_val_sp, CompilerType type,
              const std::string &name, bool use_synthetic, bool is_arrow);

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

std::string FormatDiagnostics(DILSourceManager& sm, const std::string& message,
                              uint32_t loc);

void SetUbStatus(Status& error, ErrorCode code);

/// TypeDeclaration builds information about the literal type definition as
/// type is being parsed. It doesn't perform semantic analysis for non-basic
/// types -- e.g. "char&&&" is a valid type declaration.
/// NOTE: CV qualifiers are ignored.
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

/// Pure recursive descent parser for C++ like expressions.
/// EBNF grammar for the parser is described in lldb/docs/dil-expr-lang.ebnf
class DILParser {
 public:
  explicit DILParser(std::shared_ptr<DILSourceManager> dil_sm,
                     std::shared_ptr<ExecutionContextScope> exe_ctx_scope,
                     lldb::DynamicValueType use_dynamic,
                     bool use_synthetic, bool fragile_ivar,
                     bool check_ptr_vs_member);

  DILASTNodeUP Run(Status& error);

  ~DILParser() { m_ctx_scope.reset(); }

  bool UseSynthetic() { return m_use_synthetic; }

  lldb::DynamicValueType UseDynamic() { return m_use_dynamic; }

  using PtrOperator = std::tuple<dil::TokenKind, uint32_t>;

 private:
  DILASTNodeUP ParseExpression();
  DILASTNodeUP ParseAssignmentExpression();
  DILASTNodeUP ParseLogicalOrExpression();
  DILASTNodeUP ParseLogicalAndExpression();
  DILASTNodeUP ParseInclusiveOrExpression();
  DILASTNodeUP ParseExclusiveOrExpression();
  DILASTNodeUP ParseAndExpression();
  DILASTNodeUP ParseEqualityExpression();
  DILASTNodeUP ParseRelationalExpression();
  DILASTNodeUP ParseShiftExpression();
  DILASTNodeUP ParseAdditiveExpression();
  DILASTNodeUP ParseMultiplicativeExpression();
  DILASTNodeUP ParseCastExpression();
  DILASTNodeUP ParseUnaryExpression();
  DILASTNodeUP ParsePostfixExpression();
  DILASTNodeUP ParsePrimaryExpression();

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

  bool IsSimpleTypeSpecifierKeyword(DILToken token) const;
  bool IsCvQualifier(DILToken token) const;
  bool IsPtrOperator(DILToken token) const;
  bool HandleSimpleTypeSpecifier(TypeDeclaration* type_decl);

  std::string ParseIdExpression();
  std::string ParseUnqualifiedId();
  DILASTNodeUP ParseNumericLiteral();
  DILASTNodeUP ParseBooleanLiteral();
  DILASTNodeUP ParseCharLiteral();
  DILASTNodeUP ParseStringLiteral();
  DILASTNodeUP ParsePointerLiteral();
  DILASTNodeUP ParseNumericConstant();
  DILASTNodeUP ParseFloatingLiteral(NumericLiteralParser& literal,
                                    DILToken& token);
  DILASTNodeUP ParseIntegerLiteral(NumericLiteralParser& literal,
                                   DILToken& token);
  DILASTNodeUP ParseBuiltinFunction(uint32_t loc,
                                  std::unique_ptr<BuiltinFunctionDef> func_def);

  bool ImplicitConversionIsAllowed(CompilerType src, CompilerType dst,
                                   bool is_src_literal_zero = false);
  DILASTNodeUP InsertImplicitConversion(DILASTNodeUP expr, CompilerType type);

  void ConsumeToken();

  void BailOut(ErrorCode error_code, const std::string& error,
               uint32_t loc);

  void BailOut(Status error);

  void Expect(dil::TokenKind kind);

  std::string TokenDescription(const DILToken& token);

  template <typename... Ts>
  void ExpectOneOf(dil::TokenKind k, Ts... ks);

  DILASTNodeUP BuildCStyleCast(CompilerType type, DILASTNodeUP rhs,
                             uint32_t location);
  DILASTNodeUP BuildCxxCast(dil::TokenKind kind, CompilerType type,
                          DILASTNodeUP rhs, uint32_t location);
  DILASTNodeUP BuildCxxDynamicCast(CompilerType type, DILASTNodeUP rhs,
                                 uint32_t location);
  DILASTNodeUP BuildCxxStaticCast(CompilerType type, DILASTNodeUP rhs,
                                uint32_t location);
  DILASTNodeUP BuildCxxStaticCastToScalar(CompilerType type, DILASTNodeUP rhs,
                                        uint32_t location);
  DILASTNodeUP BuildCxxStaticCastToEnum(CompilerType type, DILASTNodeUP rhs,
                                      uint32_t location);
  DILASTNodeUP BuildCxxStaticCastToPointer(CompilerType type, DILASTNodeUP rhs,
                                         uint32_t location);
  DILASTNodeUP BuildCxxStaticCastToNullPtr(CompilerType type, DILASTNodeUP rhs,
                                         uint32_t location);
  DILASTNodeUP BuildCxxStaticCastToReference(CompilerType type, DILASTNodeUP rhs,
                                           uint32_t location);
  DILASTNodeUP BuildCxxStaticCastForInheritedTypes(
      CompilerType type, DILASTNodeUP rhs, uint32_t location);
  DILASTNodeUP BuildCxxReinterpretCast(CompilerType type, DILASTNodeUP rhs,
                                     uint32_t location);
  DILASTNodeUP BuildUnaryOp(UnaryOpKind kind, DILASTNodeUP rhs,
                          uint32_t location);
  DILASTNodeUP BuildIncrementDecrement(UnaryOpKind kind, DILASTNodeUP rhs,
                                     uint32_t location);
  DILASTNodeUP BuildBinaryOp(BinaryOpKind kind, DILASTNodeUP lhs, DILASTNodeUP rhs,
                           uint32_t location);
  CompilerType PrepareBinaryAddition(DILASTNodeUP& lhs, DILASTNodeUP& rhs,
                                     uint32_t location,
                                     bool is_comp_assign);
  CompilerType PrepareBinarySubtraction(DILASTNodeUP& lhs, DILASTNodeUP& rhs,
                                        uint32_t location,
                                        bool is_comp_assign);
  CompilerType PrepareBinaryMulDiv(DILASTNodeUP& lhs, DILASTNodeUP& rhs,
                                   bool is_comp_assign);
  CompilerType PrepareBinaryRemainder(DILASTNodeUP& lhs, DILASTNodeUP& rhs,
                                      bool is_comp_assign);
  CompilerType PrepareBinaryBitwise(DILASTNodeUP& lhs, DILASTNodeUP& rhs,
                                    bool is_comp_assign);
  CompilerType PrepareBinaryShift(DILASTNodeUP& lhs, DILASTNodeUP& rhs,
                                  bool is_comp_assign);
  CompilerType PrepareBinaryComparison(BinaryOpKind kind, DILASTNodeUP& lhs,
                                 DILASTNodeUP& rhs,
                                 uint32_t location);
  CompilerType PrepareBinaryLogical(const DILASTNodeUP& lhs,
                                    const DILASTNodeUP& rhs);
  DILASTNodeUP BuildBinarySubscript(DILASTNodeUP lhs, DILASTNodeUP rhs,
                                  uint32_t location);
  CompilerType PrepareCompositeAssignment(CompilerType comp_assign_type,
                                          const DILASTNodeUP& lhs,
                                          uint32_t location);
  DILASTNodeUP BuildTernaryOp(DILASTNodeUP cond, DILASTNodeUP lhs, DILASTNodeUP rhs,
                            uint32_t location);
  DILASTNodeUP BuildMemberOf(DILASTNodeUP lhs, std::string member_id,
                            bool is_arrow,
                            uint32_t location);

  bool AllowSideEffects() const { return m_allow_side_effects; }

  void SetAllowSideEffects (bool allow_side_effects) {
    m_allow_side_effects = allow_side_effects;
  }

  const IdentifierInfo& GetInfo(const IdentifierNode *node) {
    return node->info();
  }

  void TentativeParsingRollback(uint32_t saved_idx) {
    m_error.Clear();
    m_dil_lexer.ResetTokenIdx(saved_idx);
    m_dil_token = m_dil_lexer.GetCurrentToken();
  }

  // Parser doesn't own the evaluation context. The produced AST may depend on
  // it (for example, for source locations), so it's expected that expression
  // context will outlive the parser.
  std::shared_ptr<ExecutionContextScope> m_ctx_scope;

  std::shared_ptr<DILSourceManager> m_sm;
  // The token lexer is stopped at (aka "current token").
  DILToken m_dil_token;
  // Holds an error if it occures during parsing.
  Status m_error;

  bool m_allow_side_effects = true;

  lldb::DynamicValueType m_use_dynamic;
  bool m_use_synthetic;
  bool m_fragile_ivar;
  bool m_check_ptr_vs_member;
  DILLexer m_dil_lexer;
}; // class DILParser

}  // namespace dil

}  // namespace lldb_private

#endif  // LLDB_VALUEOBJECT_DILPARSER_H_
