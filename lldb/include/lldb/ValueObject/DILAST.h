//===-- DILAST.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_VALUEOBJECT_DILAST_H
#define LLDB_VALUEOBJECT_DILAST_H

#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/TypeList.h"
#include "lldb/Target/LanguageRuntime.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/ValueObject/ValueObject.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/TokenKinds.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Casting.h"

namespace lldb_private {

namespace dil {

/// Struct to hold information about member fields. Used by the parser for the
/// Data Inspection Language (DIL).
struct MemberInfo {
  std::optional<std::string> name;
  CompilerType type;
  std::optional<uint32_t> bitfield_size_in_bits;
  bool is_synthetic;
  bool is_dynamic;
  lldb::ValueObjectSP val_obj_sp;
};

/// Get the appropriate ValueObjectSP, consulting the use_dynamic and
/// use_synthetic options passed.
lldb::ValueObjectSP
GetDynamicOrSyntheticValue(lldb::ValueObjectSP valobj_sp,
                           lldb::DynamicValueType use_dynamic = lldb::eNoDynamicValues,
                           bool use_synthetic = false);

/// The various types DIL AST nodes (used by the DIL parser).
enum class NodeKind {
  eErrorNode,
  eScalarLiteralNode,
  eStringLiteralNode,
  eIdentifierNode,
  eSizeOfNode,
  eBuiltinFunctionCallNode,
  eCStyleCastNode,
  eCxxStaticCastNode,
  eCxxReinterpretCastNode,
  eMemberOfNode,
  eArraySubscriptNode,
  eBinaryOpNode,
  eUnaryOpNode,
  eTernaryOpNode,
  //eSmartPtrToPtrDecayNode
};


/// Type promotion cast kinds in DIL.
enum class TypePromotionCastKind {
  eArithmetic,
  ePointer,
  eNone,
};

/// The C-Style casts allowed by DIL.
enum class CStyleCastKind {
  eEnumeration,
  eNullptr,
  eReference,
  eNone,
};

/// The Cxx static casts allowed by DIL.
enum class CxxStaticCastKind {
  eNoOp,
  //eArithmetic,
  eEnumeration,
  //ePointer,
  eNullptr,
  eBaseToDerived,
  eDerivedToBase,
  eNone,
};

/// The binary operators recognized by DIL.
enum class BinaryOpKind {
  Mul,       // "*"
  Div,       // "/"
  Rem,       // "%"
  Add,       // "+"
  Sub,       // "-"
  Shl,       // "<<"
  Shr,       // ">>"
  LT,        // "<"
  GT,        // ">"
  LE,        // "<="
  GE,        // ">="
  EQ,        // "=="
  NE,        // "!="
  And,       // "&"
  Xor,       // "^"
  Or,        // "|"
  LAnd,      // "&&"
  LOr,       // "||"
  Assign,    // "="
  MulAssign, // "*="
  DivAssign, // "/="
  RemAssign, // "%="
  AddAssign, // "+="
  SubAssign, // "-="
  ShlAssign, // "<<="
  ShrAssign, // ">>="
  AndAssign, // "&="
  XorAssign, // "^="
  OrAssign,  // "|="
};

/// The Unary operators recognized by DIL.
enum class UnaryOpKind {
  PostInc, // "++"
  PostDec, // "--"
  PreInc,  // "++"
  PreDec,  // "--"
  AddrOf,  // "&"
  Deref,   // "*"
  Plus,    // "+"
  Minus,   // "-"
  Not,     // "~"
  LNot,    // "!"
};

/// Helper functions for DIL AST node parsing.

/// Translates clang tokens to BinaryOpKind.
BinaryOpKind
clang_token_kind_to_binary_op_kind(clang::tok::TokenKind token_kind);

/// Returns bool indicating whether or not the input kind is an assignment.
bool binary_op_kind_is_comp_assign(BinaryOpKind kind);

/// Given a string representing a type, returns the CompilerType corresponding
/// to the named type, if it exists.
CompilerType
ResolveTypeByName(const std::string &name,
                  std::shared_ptr<ExecutionContextScope> ctx_scope);

/// Class used to store & manipulate information about identifiers.
class IdentifierInfo {
public:
  enum class Kind {
    eValue,
    eContextArg,
    eMemberPath,
  };

  static std::unique_ptr<IdentifierInfo> FromValue(ValueObject &valobj) {
    CompilerType type;
    type = valobj.GetCompilerType();
    return std::unique_ptr<IdentifierInfo>(
        new IdentifierInfo(Kind::eValue, type, valobj.GetSP(), {}));
  }

  static std::unique_ptr<IdentifierInfo> FromContextArg(CompilerType type) {
    lldb::ValueObjectSP empty_value;
    return std::unique_ptr<IdentifierInfo>(
        new IdentifierInfo(Kind::eContextArg, type, empty_value, {}));
  }

  static std::unique_ptr<IdentifierInfo>
  FromMemberPath(CompilerType type, std::vector<uint32_t> path) {
    lldb::ValueObjectSP empty_value;
    return std::unique_ptr<IdentifierInfo>(new IdentifierInfo(
        Kind::eMemberPath, type, empty_value, std::move(path)));
  }

  Kind GetKind() const { return m_kind; }
  lldb::ValueObjectSP GetValue() const { return m_value; }
  const std::vector<uint32_t> &GetPath() const { return m_path; }

  CompilerType GetType() { return m_type; }
  bool IsValid() const { return m_type.IsValid(); }

  IdentifierInfo(Kind kind, CompilerType type, lldb::ValueObjectSP value,
                 std::vector<uint32_t> path)
      : m_kind(kind), m_type(type), m_value(std::move(value)),
        m_path(std::move(path)) {}

private:
  Kind m_kind;
  CompilerType m_type;
  lldb::ValueObjectSP m_value;
  std::vector<uint32_t> m_path;
};

/// Given the name of an identifier (variable name, member name, type name,
/// etc.), find the ValueObject for that name (if it exists) and create and
/// return an IdentifierInfo object containing all the relevant information
/// about that object (for DIL parsing and evaluating).
std::unique_ptr<IdentifierInfo> LookupIdentifier(
    const std::string &name, std::shared_ptr<ExecutionContextScope> ctx_scope,
    lldb::DynamicValueType use_dynamic, CompilerType *scope_ptr = nullptr);

/// Forward declaration, for use in DIL AST nodes. Definition is at the very
/// end of this file.
class Visitor;

/// The rest of the classes in this file, except for the Visitor class at the
/// very end, define all the types of AST nodes used by the DIL parser and
/// expression evaluator. The DIL parser parses the input string and creates the
/// AST parse tree from the AST nodes. The resulting AST node tree gets passed
/// to the DIL expression evaluator, which evaluates the DIL AST nodes and
/// creates/returns a ValueObjectSP containing the result.

/// Base class for AST nodes used by the Data Inspection Language (DIL) parser.
/// All of the specialized types of AST nodes inherit from this (virtual) base
/// class.
class DILASTNode {
public:
  DILASTNode(clang::SourceLocation location, NodeKind kind) :
      m_location(location), m_kind(kind) {}
  virtual ~DILASTNode() = default;

  virtual void Accept(Visitor *v) const = 0;

  virtual bool is_rvalue() const = 0;
  virtual bool is_bitfield() const { return false; };
  virtual bool is_context_var() const { return false; };
  virtual bool is_literal_zero() const { return false; }
  virtual uint32_t bitfield_size() const { return 0; }
  virtual CompilerType result_type() const = 0;
  virtual ValueObject *valobj() const { return nullptr; }

  clang::SourceLocation GetLocation() const { return m_location; }
  NodeKind GetKind() const { return m_kind; }

  // The expression result type, but dereferenced in case it's a reference. This
  // is for convenience, since for the purposes of the semantic analysis only
  // the dereferenced type matters.
  CompilerType GetDereferencedResultType() const;

private:
  clang::SourceLocation m_location;
  const NodeKind m_kind;
};

using DILASTNodeUP = std::unique_ptr<DILASTNode>;

class ErrorNode : public DILASTNode {
public:
  ErrorNode(CompilerType empty_type)
      : DILASTNode(clang::SourceLocation(), NodeKind::eErrorNode),
        m_empty_type(empty_type) {}
  void Accept(Visitor *v) const override;
  bool is_rvalue() const override { return false; }
  CompilerType result_type() const override { return m_empty_type; }
  CompilerType result_type_real() const { return m_empty_type; }

  static bool classof(const DILASTNode *node) {
    return node->GetKind() == NodeKind::eErrorNode;
  }

private:
  CompilerType m_empty_type;
};

class ScalarLiteralNode : public DILASTNode {
public:
  ScalarLiteralNode(clang::SourceLocation location, CompilerType type,
                    Scalar value)
      : DILASTNode(location, NodeKind::eScalarLiteralNode), m_type(type),
        m_value(value) {}

  void Accept(Visitor *v) const override;
  bool is_rvalue() const override { return true; }
  bool is_literal_zero() const override {
    return m_value.IsZero() && !m_type.IsBoolean();
  }
  CompilerType result_type() const override { return m_type; }

  Scalar GetValue() const & { return m_value; }

  static bool classof(const DILASTNode *node) {
    return node->GetKind() == NodeKind::eScalarLiteralNode;
  }

private:
  CompilerType m_type;
  Scalar m_value;
};

class StringLiteralNode : public DILASTNode {
public:
  StringLiteralNode(clang::SourceLocation location, CompilerType type,
              std::string value)
      : DILASTNode(location, NodeKind::eStringLiteralNode),
        m_type(type),
        m_value(value) {}

  void Accept(Visitor *v) const override;
  bool is_rvalue() const override { return true; }
  CompilerType result_type() const override { return m_type; }

  std::string GetValue() const & { return m_value; }

  static bool classof(const DILASTNode *node) {
    return node->GetKind() == NodeKind::eStringLiteralNode;
  }

private:
  CompilerType m_type;
  std::string m_value;
};

class IdentifierNode : public DILASTNode {
public:
  IdentifierNode(clang::SourceLocation location, std::string name,
                 std::unique_ptr<IdentifierInfo> identifier, bool is_rvalue,
                 bool is_context_var)
      : DILASTNode(location, NodeKind::eIdentifierNode),
        m_is_rvalue(is_rvalue),
        m_is_context_var(is_context_var), m_name(std::move(name)),
        m_identifier(std::move(identifier)) {}

  void Accept(Visitor *v) const override;
  bool is_rvalue() const override { return m_is_rvalue; }
  bool is_context_var() const override { return m_is_context_var; };
  CompilerType result_type() const override { return m_identifier->GetType(); }
  ValueObject *valobj() const override {
    return m_identifier->GetValue().get();
  }

  std::string name() const { return m_name; }
  const IdentifierInfo &info() const { return *m_identifier; }

  static bool classof(const DILASTNode *node) {
    return node->GetKind() == NodeKind::eIdentifierNode;
  }

private:
  bool m_is_rvalue;
  bool m_is_context_var;
  std::string m_name;
  std::unique_ptr<IdentifierInfo> m_identifier;
};

class SizeOfNode : public DILASTNode {
public:
  SizeOfNode(clang::SourceLocation location, CompilerType type,
             CompilerType operand)
      : DILASTNode(location, NodeKind::eSizeOfNode),
        m_type(type), m_operand(operand) {}

  void Accept(Visitor *v) const override;
  bool is_rvalue() const override { return true; }
  CompilerType result_type() const override { return m_type; }

  CompilerType operand() const { return m_operand; }

  static bool classof(const DILASTNode *node) {
    return node->GetKind() == NodeKind::eSizeOfNode;
  }

private:
  CompilerType m_type;
  CompilerType m_operand;
};

class BuiltinFunctionCallNode : public DILASTNode {
public:
  BuiltinFunctionCallNode(clang::SourceLocation location,
                          CompilerType result_type, std::string name,
                          std::vector<DILASTNodeUP> arguments)
      : DILASTNode(location, NodeKind::eBuiltinFunctionCallNode),
        m_result_type(result_type),
        m_name(std::move(name)), m_arguments(std::move(arguments)) {}

  void Accept(Visitor *v) const override;
  bool is_rvalue() const override { return true; }
  CompilerType result_type() const override { return m_result_type; }

  std::string name() const { return m_name; }
  const std::vector<DILASTNodeUP> &arguments() const { return m_arguments; };

  static bool classof(const DILASTNode *node) {
    return node->GetKind() == NodeKind::eBuiltinFunctionCallNode;
  }

private:
  CompilerType m_result_type;
  std::string m_name;
  std::vector<DILASTNodeUP> m_arguments;
};

class CStyleCastNode : public DILASTNode {
public:
  CStyleCastNode(clang::SourceLocation location, CompilerType type,
                 DILASTNodeUP operand, CStyleCastKind kind)
      : DILASTNode(location, NodeKind::eCStyleCastNode),
        m_type(type), m_operand(std::move(operand)),
        m_cast_kind(kind) { m_promo_kind = TypePromotionCastKind::eNone; }

  CStyleCastNode(clang::SourceLocation location, CompilerType type,
                 DILASTNodeUP operand, TypePromotionCastKind kind)
      : DILASTNode(location, NodeKind::eCStyleCastNode),
        m_type(type), m_operand(std::move(operand)),
        m_promo_kind(kind) { m_cast_kind = CStyleCastKind::eNone; }

  void Accept(Visitor *v) const override;
  bool is_rvalue() const override {
    return m_cast_kind != CStyleCastKind::eReference;
  }
  CompilerType result_type() const override { return m_type; }
  ValueObject *valobj() const override { return m_operand->valobj(); }

  CompilerType type() const { return m_type; }
  DILASTNode *operand() const { return m_operand.get(); }
  CStyleCastKind cast_kind() const { return m_cast_kind; }
  TypePromotionCastKind promo_kind() const { return m_promo_kind; }

  static bool classof(const DILASTNode *node) {
    return node->GetKind() == NodeKind::eCStyleCastNode;
  }

private:
  CompilerType m_type;
  DILASTNodeUP m_operand;
  CStyleCastKind m_cast_kind;
  TypePromotionCastKind m_promo_kind;
};

class CxxStaticCastNode : public DILASTNode {
public:
  CxxStaticCastNode(clang::SourceLocation location, CompilerType type,
                    DILASTNodeUP operand, CxxStaticCastKind kind,
                    bool is_rvalue)
      : DILASTNode(location, NodeKind::eCxxStaticCastNode),
        m_type(type), m_operand(std::move(operand)), m_cast_kind(kind),
        m_is_rvalue(is_rvalue) {
    assert(kind != CxxStaticCastKind::eBaseToDerived &&
           kind != CxxStaticCastKind::eDerivedToBase &&
           "invalid constructor for base-to-derived and derived-to-base casts");
    m_promo_kind = TypePromotionCastKind::eNone;
  }

  CxxStaticCastNode(clang::SourceLocation location, CompilerType type,
                    DILASTNodeUP operand, TypePromotionCastKind kind,
                    bool is_rvalue)
      : DILASTNode(location, NodeKind::eCxxStaticCastNode),
        m_type(type), m_operand(std::move(operand)), m_promo_kind(kind),
        m_is_rvalue(is_rvalue) {
    m_cast_kind = CxxStaticCastKind::eNone;
  }

  CxxStaticCastNode(clang::SourceLocation location, CompilerType type,
                    DILASTNodeUP operand, std::vector<uint32_t> idx, bool is_rvalue)
      : DILASTNode(location, NodeKind::eCxxStaticCastNode),
        m_type(type), m_operand(std::move(operand)),
        m_idx(std::move(idx)), m_cast_kind(CxxStaticCastKind::eDerivedToBase),
        m_is_rvalue(is_rvalue) {
    m_promo_kind = TypePromotionCastKind::eNone;
  }

  CxxStaticCastNode(clang::SourceLocation location, CompilerType type,
                    DILASTNodeUP operand, uint64_t offset, bool is_rvalue)
      : DILASTNode(location, NodeKind::eCxxStaticCastNode),
        m_type(type), m_operand(std::move(operand)),
        m_offset(offset), m_cast_kind(CxxStaticCastKind::eBaseToDerived),
        m_is_rvalue(is_rvalue) {
    m_promo_kind = TypePromotionCastKind::eNone;
  }

  void Accept(Visitor *v) const override;
  bool is_rvalue() const override { return m_is_rvalue; }
  CompilerType result_type() const override { return m_type; }
  ValueObject *valobj() const override { return m_operand->valobj(); }

  CompilerType type() const { return m_type; }
  DILASTNode *operand() const { return m_operand.get(); }
  const std::vector<uint32_t> &idx() const { return m_idx; }
  uint64_t offset() const { return m_offset; }
  CxxStaticCastKind cast_kind() const { return m_cast_kind; }
  TypePromotionCastKind promo_kind() const { return m_promo_kind; }

  static bool classof(const DILASTNode *node) {
    return node->GetKind() == NodeKind::eCxxStaticCastNode;
  }

private:
  CompilerType m_type;
  DILASTNodeUP m_operand;
  std::vector<uint32_t> m_idx;
  uint64_t m_offset = 0;
  CxxStaticCastKind m_cast_kind;
  TypePromotionCastKind m_promo_kind;
  bool m_is_rvalue;
};

class CxxReinterpretCastNode : public DILASTNode {
public:
  CxxReinterpretCastNode(clang::SourceLocation location, CompilerType type,
                         DILASTNodeUP operand, bool is_rvalue)
      : DILASTNode(location, NodeKind::eCxxReinterpretCastNode),
        m_type(type), m_operand(std::move(operand)),
        m_is_rvalue(is_rvalue) {}

  void Accept(Visitor *v) const override;
  bool is_rvalue() const override { return m_is_rvalue; }
  CompilerType result_type() const override { return m_type; }
  ValueObject *valobj() const override { return m_operand->valobj(); }

  CompilerType type() const { return m_type; }
  DILASTNode *operand() const { return m_operand.get(); }

  static bool classof(const DILASTNode *node) {
    return node->GetKind() == NodeKind::eCxxReinterpretCastNode;
  }

private:
  CompilerType m_type;
  DILASTNodeUP m_operand;
  bool m_is_rvalue;
};

class MemberOfNode : public DILASTNode {
public:
  MemberOfNode(clang::SourceLocation location, CompilerType result_type,
               DILASTNodeUP base, std::optional<uint32_t> bitfield_size,
               std::vector<uint32_t> member_index, bool is_arrow,
               bool is_synthetic, bool is_dynamic, ConstString name,
               lldb::ValueObjectSP field_valobj_sp)
      : DILASTNode(location, NodeKind::eMemberOfNode),
        m_result_type(result_type), m_base(std::move(base)),
        m_bitfield_size(bitfield_size), m_member_index(std::move(member_index)),
        m_is_arrow(is_arrow), m_is_synthetic(is_synthetic),
        m_is_dynamic(is_dynamic), m_field_name(name),
        m_field_valobj_sp(field_valobj_sp) {}

  void Accept(Visitor *v) const override;
  bool is_rvalue() const override { return false; }
  bool is_bitfield() const override { return m_bitfield_size ? true : false; }
  uint32_t bitfield_size() const override {
    return m_bitfield_size ? m_bitfield_size.value() : 0;
  }
  CompilerType result_type() const override { return m_result_type; }
  ValueObject *valobj() const override { return m_field_valobj_sp.get(); }

  DILASTNode *base() const { return m_base.get(); }
  const std::vector<uint32_t> &member_index() const { return m_member_index; }
  bool is_arrow() const { return m_is_arrow; }
  bool is_synthetic() const { return m_is_synthetic; }
  bool is_dynamic() const { return m_is_dynamic; }
  ConstString field_name() const { return m_field_name; }

  static bool classof(const DILASTNode *node) {
    return node->GetKind() == NodeKind::eMemberOfNode;
  }

private:
  CompilerType m_result_type;
  DILASTNodeUP m_base;
  std::optional<uint32_t> m_bitfield_size;
  std::vector<uint32_t> m_member_index;
  bool m_is_arrow;
  bool m_is_synthetic;
  bool m_is_dynamic;
  ConstString m_field_name;
  lldb::ValueObjectSP m_field_valobj_sp;
};

class ArraySubscriptNode : public DILASTNode {
public:
  ArraySubscriptNode(clang::SourceLocation location, CompilerType result_type,
                     DILASTNodeUP base, DILASTNodeUP index)
      : DILASTNode(location, NodeKind::eArraySubscriptNode),
        m_result_type(result_type),
        m_base(std::move(base)), m_index(std::move(index)) {}

  void Accept(Visitor *v) const override;
  bool is_rvalue() const override { return false; }
  CompilerType result_type() const override { return m_result_type; }
  ValueObject *valobj() const override {
    ValueObject *base_obj = m_base->valobj();
    ValueObject *idx_obj = m_index->valobj();
    Status error;
    int idx = 0;
    if (idx_obj && idx_obj->GetCompilerType().IsReferenceType())
      idx_obj = idx_obj->Dereference(error).get();
    if (idx_obj && error.Success())
      idx = idx_obj->GetValueAsUnsigned(0);
    if (base_obj->GetChildAtIndex(idx))
      return (base_obj->GetChildAtIndex(idx)).get();
    return base_obj;
  }

  DILASTNode *base() const { return m_base.get(); }
  DILASTNode *index() const { return m_index.get(); }

  static bool classof(const DILASTNode *node) {
    return node->GetKind() == NodeKind::eArraySubscriptNode;
  }

private:
  CompilerType m_result_type;
  DILASTNodeUP m_base;
  DILASTNodeUP m_index;
};

class BinaryOpNode : public DILASTNode {
public:
  BinaryOpNode(clang::SourceLocation location, CompilerType result_type,
               BinaryOpKind kind, DILASTNodeUP lhs, DILASTNodeUP rhs,
               CompilerType comp_assign_type,
               ValueObject *val_obj_ptr = nullptr)
      : DILASTNode(location, NodeKind::eBinaryOpNode),
        m_result_type(result_type), m_kind(kind),
        m_lhs(std::move(lhs)), m_rhs(std::move(rhs)),
        m_comp_assign_type(comp_assign_type) {
      if (val_obj_ptr)
      m_val_obj_sp = val_obj_ptr->GetSP();
  }

  void Accept(Visitor *v) const override;
  bool is_rvalue() const override {
    return !binary_op_kind_is_comp_assign(m_kind);
  }
  CompilerType result_type() const override { return m_result_type; }
  ValueObject *valobj() const override {
    if (m_val_obj_sp)
      return m_val_obj_sp.get();
    ValueObject *rhs_valobj = m_rhs->valobj();
    ValueObject *lhs_valobj = m_lhs->valobj();
    if (lhs_valobj)
      return lhs_valobj;
    if (rhs_valobj)
      return rhs_valobj;
    return m_val_obj_sp.get();
  }

  BinaryOpKind kind() const { return m_kind; }
  DILASTNode *lhs() const { return m_lhs.get(); }
  DILASTNode *rhs() const { return m_rhs.get(); }
  CompilerType comp_assign_type() const { return m_comp_assign_type; }

  static bool classof(const DILASTNode *node) {
    return node->GetKind() == NodeKind::eBinaryOpNode;
  }

private:
  CompilerType m_result_type;
  BinaryOpKind m_kind;
  DILASTNodeUP m_lhs;
  DILASTNodeUP m_rhs;
  CompilerType m_comp_assign_type;
  lldb::ValueObjectSP m_val_obj_sp;
};

class UnaryOpNode : public DILASTNode {
public:
  UnaryOpNode(clang::SourceLocation location, CompilerType result_type,
              UnaryOpKind kind, DILASTNodeUP rhs)
      : DILASTNode(location, NodeKind::eUnaryOpNode),
        m_result_type(result_type), m_kind(kind),
        m_rhs(std::move(rhs)) {}

  void Accept(Visitor *v) const override;
  bool is_rvalue() const override { return m_kind != UnaryOpKind::Deref; }
  CompilerType result_type() const override { return m_result_type; }
  ValueObject *valobj() const override { return m_rhs->valobj(); }

  UnaryOpKind kind() const { return m_kind; }
  DILASTNode *rhs() const { return m_rhs.get(); }

  static bool classof(const DILASTNode *node) {
    return node->GetKind() == NodeKind::eUnaryOpNode;
  }

private:
  CompilerType m_result_type;
  UnaryOpKind m_kind;
  DILASTNodeUP m_rhs;
};

class TernaryOpNode : public DILASTNode {
public:
  TernaryOpNode(clang::SourceLocation location, CompilerType result_type,
                DILASTNodeUP cond, DILASTNodeUP lhs, DILASTNodeUP rhs)
      : DILASTNode(location, NodeKind::eTernaryOpNode),
        m_result_type(result_type),
        m_cond(std::move(cond)), m_lhs(std::move(lhs)), m_rhs(std::move(rhs)) {}

  void Accept(Visitor *v) const override;
  bool is_rvalue() const override {
    return m_lhs->is_rvalue() || m_rhs->is_rvalue();
  }
  bool is_bitfield() const override {
    return m_lhs->is_bitfield() || m_rhs->is_bitfield();
  }
  CompilerType result_type() const override { return m_result_type; }

  DILASTNode *cond() const { return m_cond.get(); }
  DILASTNode *lhs() const { return m_lhs.get(); }
  DILASTNode *rhs() const { return m_rhs.get(); }

  static bool classof(const DILASTNode *node) {
    return node->GetKind() == NodeKind::eTernaryOpNode;
  }

private:
  CompilerType m_result_type;
  DILASTNodeUP m_cond;
  DILASTNodeUP m_lhs;
  DILASTNodeUP m_rhs;
};


//class SmartPtrToPtrDecay : public DILASTNode {
//public:
//  SmartPtrToPtrDecay(clang::SourceLocation location, CompilerType result_type,
//                     DILASTNodeUP ptr)
//      : DILASTNode(location, NodeKind::eSmartPtrToPtrDecayNode),
//        m_result_type(result_type),
//        m_ptr(std::move(ptr)) {}
//
//  void Accept(Visitor *v) const override;
//  bool is_rvalue() const override { return false; }
//  CompilerType result_type() const override { return m_result_type; }
//
//  DILASTNode *ptr() const { return m_ptr.get(); }
//
//  static bool classof(const DILASTNode *node) {
//    return node->GetKind() == NodeKind::eSmartPtrToPtrDecayNode;
//  }
//
//private:
//  CompilerType m_result_type;
//  DILASTNodeUP m_ptr;
//};

/// This class contains one Visit method for each specialized type of
/// DIL AST node. The Visit methods are used to dispatch a DIL AST node to
/// the correct function in the DIL expression evaluator for evaluating that
/// type of AST node.
class Visitor {
public:
  virtual ~Visitor() = default;
  virtual void Visit(const ErrorNode *node) = 0;
  virtual void Visit(const ScalarLiteralNode *node) = 0;
  virtual void Visit(const StringLiteralNode *node) = 0;
  virtual void Visit(const IdentifierNode *node) = 0;
  virtual void Visit(const SizeOfNode *node) = 0;
  virtual void Visit(const BuiltinFunctionCallNode *node) = 0;
  virtual void Visit(const CStyleCastNode *node) = 0;
  virtual void Visit(const CxxStaticCastNode *node) = 0;
  virtual void Visit(const CxxReinterpretCastNode *node) = 0;
  virtual void Visit(const MemberOfNode *node) = 0;
  virtual void Visit(const ArraySubscriptNode *node) = 0;
  virtual void Visit(const BinaryOpNode *node) = 0;
  virtual void Visit(const UnaryOpNode *node) = 0;
  virtual void Visit(const TernaryOpNode *node) = 0;
  // virtual void Visit(const SmartPtrToPtrDecay *node) = 0;
};

}  // namespace dil

} // namespace lldb_private

#endif // LLDB_VALUEOBJECT_DILAST_H
