//===-- DILAST.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_CORE_DILAST_H
#define LLDB_CORE_DILAST_H

#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "lldb/Core/ValueObject.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/TypeList.h"
#include "lldb/Target/LanguageRuntime.h"
#include "lldb/Utility/ConstString.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/TokenKinds.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Casting.h"

namespace lldb_private {

namespace DIL {

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
/// use_synthetic options passed, acquiring the process & target locks if
/// appropriate.
lldb::ValueObjectSP GetDynamicOrSyntheticValue(
    lldb::ValueObjectSP valobj_sp,
    lldb::DynamicValueType use_dynamic = lldb::eNoDynamicValues,
    bool use_synthetic = false);

/// The various types DIL AST nodes (used by the DIL parser).
enum class NodeKind {
  eErrorNode,
  eScalarLiteralNode,
  eStringLiteralNode,
  eIdentifierNode,
  eCStyleCastNode,
  eMemberOfNode,
  eArraySubscriptNode,
  eUnaryOpNode,
  eSmartPtrToPtrDecayNode
};

/// The C-Style casts for type promotion allowed by DIL.
enum class TypePromotionCastKind {
  eArithmetic,
  ePointer,
};

/// The Unary operators recognized by DIL.
enum class UnaryOpKind {
  AddrOf, // "&"
  Deref,  // "*"
  Minus,  // "-"
};

/// Given a string representing a type, returns the CompilerType corresponding
/// to the named type, if it exists.
CompilerType
ResolveTypeByName(const std::string &name,
                  std::shared_ptr<ExecutionContextScope> ctx_scope);

/// Class used to store & manipulate information about identifiers.
class IdentifierInfo {
private:
  using MemberPath = std::vector<uint32_t>;
  using IdentifierInfoUP = std::unique_ptr<IdentifierInfo>;

public:
  enum class Kind {
    eValue,
    eContextArg,
    eMemberPath,
    eThisKeyword,
  };

  static IdentifierInfoUP FromValue(lldb::ValueObjectSP value_sp) {
    CompilerType type;
    if (value_sp)
      type = value_sp->GetCompilerType();
    return IdentifierInfoUP(
        new IdentifierInfo(Kind::eValue, type, value_sp, {}));
  }

  static IdentifierInfoUP FromContextArg(CompilerType type) {
    lldb::ValueObjectSP empty_value;
    return IdentifierInfoUP(
        new IdentifierInfo(Kind::eContextArg, type, empty_value, {}));
  }

  static IdentifierInfoUP FromMemberPath(CompilerType type, MemberPath path) {
    lldb::ValueObjectSP empty_value;
    return IdentifierInfoUP(new IdentifierInfo(Kind::eMemberPath, type,
                                               empty_value, std::move(path)));
  }

  static IdentifierInfoUP FromThisKeyword(CompilerType type) {
    lldb::ValueObjectSP empty_value;
    return IdentifierInfoUP(
        new IdentifierInfo(Kind::eThisKeyword, type, empty_value, {}));
  }

  Kind kind() const { return m_kind; }
  lldb::ValueObjectSP value() const { return m_value; }
  const MemberPath &path() const { return m_path; }

  CompilerType GetType() { return m_type; }
  bool IsValid() const { return m_type.IsValid(); }

  IdentifierInfo(Kind kind, CompilerType type, lldb::ValueObjectSP value,
                 MemberPath path)
      : m_kind(kind), m_type(type), m_value(std::move(value)),
        m_path(std::move(path)) {}

private:
  Kind m_kind;
  CompilerType m_type;
  lldb::ValueObjectSP m_value;
  MemberPath m_path;
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
  DILASTNode(clang::SourceLocation location, NodeKind kind)
      : m_location(location), m_kind(kind) {}
  virtual ~DILASTNode() = default;

  virtual void Accept(Visitor *v) const = 0;

  virtual bool is_rvalue() const = 0;
  virtual bool is_bitfield() const { return false; };
  virtual bool is_context_var() const { return false; };
  virtual bool is_literal_zero() const { return false; }
  virtual uint32_t bitfield_size() const { return 0; }
  virtual CompilerType result_type() const = 0;

  clang::SourceLocation GetLocation() const { return m_location; }
  NodeKind getKind() const { return m_kind; }

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
    return node->getKind() == NodeKind::eErrorNode;
  }

private:
  CompilerType m_empty_type;
};

class ScalarLiteralNode : public DILASTNode {
public:
  ScalarLiteralNode(clang::SourceLocation location, CompilerType type,
                    Scalar value, bool is_literal_zero)
      : DILASTNode(location, NodeKind::eScalarLiteralNode), m_type(type),
        m_value(value), m_is_literal_zero(is_literal_zero) {}

  void Accept(Visitor *v) const override;
  bool is_rvalue() const override { return true; }
  bool is_literal_zero() const override { return m_is_literal_zero; }
  CompilerType result_type() const override { return m_type; }

  auto value() const { return m_value; }

  static bool classof(const DILASTNode *node) {
    return node->getKind() == NodeKind::eScalarLiteralNode;
  }

private:
  CompilerType m_type;
  Scalar m_value;
  bool m_is_literal_zero;
};

class StringLiteralNode : public DILASTNode {
public:
  StringLiteralNode(clang::SourceLocation location, CompilerType type,
                    std::vector<char> value, bool is_literal_zero)
      : DILASTNode(location, NodeKind::eStringLiteralNode), m_type(type),
        m_value(value), m_is_literal_zero(is_literal_zero) {}

  void Accept(Visitor *v) const override;
  bool is_rvalue() const override { return true; }
  bool is_literal_zero() const override { return m_is_literal_zero; }
  CompilerType result_type() const override { return m_type; }

  auto value() const { return m_value; }

  static bool classof(const DILASTNode *node) {
    return node->getKind() == NodeKind::eStringLiteralNode;
  }

private:
  CompilerType m_type;
  std::vector<char> m_value;
  bool m_is_literal_zero;
};

class IdentifierNode : public DILASTNode {
public:
  IdentifierNode(clang::SourceLocation location, std::string name,
                 std::unique_ptr<IdentifierInfo> identifier, bool is_rvalue,
                 bool is_context_var)
      : DILASTNode(location, NodeKind::eIdentifierNode), m_is_rvalue(is_rvalue),
        m_is_context_var(is_context_var), m_name(std::move(name)),
        m_identifier(std::move(identifier)) {}

  void Accept(Visitor *v) const override;
  bool is_rvalue() const override { return m_is_rvalue; }
  bool is_context_var() const override { return m_is_context_var; };
  CompilerType result_type() const override { return m_identifier->GetType(); }

  std::string name() const { return m_name; }
  const IdentifierInfo &info() const { return *m_identifier; }

  static bool classof(const DILASTNode *node) {
    return node->getKind() == NodeKind::eIdentifierNode;
  }

private:
  bool m_is_rvalue;
  bool m_is_context_var;
  std::string m_name;
  std::unique_ptr<IdentifierInfo> m_identifier;
};

class CStyleCastNode : public DILASTNode {
public:
  CStyleCastNode(clang::SourceLocation location, CompilerType type,
                 DILASTNodeUP rhs, TypePromotionCastKind kind)
      : DILASTNode(location, NodeKind::eCStyleCastNode), m_type(type),
        m_rhs(std::move(rhs)), m_promo_kind(kind) {}

  void Accept(Visitor *v) const override;
  bool is_rvalue() const override { return false; }
  CompilerType result_type() const override { return m_type; }

  CompilerType type() const { return m_type; }
  DILASTNode *rhs() const { return m_rhs.get(); }
  TypePromotionCastKind promo_kind() const { return m_promo_kind; }

  static bool classof(const DILASTNode *node) {
    return node->getKind() == NodeKind::eCStyleCastNode;
  }

private:
  CompilerType m_type;
  DILASTNodeUP m_rhs;
  TypePromotionCastKind m_promo_kind;
};

class MemberOfNode : public DILASTNode {
public:
  MemberOfNode(clang::SourceLocation location, CompilerType result_type,
               DILASTNodeUP lhs, std::optional<uint32_t> bitfield_size,
               std::vector<uint32_t> member_index, bool is_arrow,
               bool is_synthetic, bool is_dynamic, ConstString name,
               lldb::ValueObjectSP valobj_sp)
      : DILASTNode(location, NodeKind::eMemberOfNode),
        m_result_type(result_type), m_lhs(std::move(lhs)),
        m_member_index(std::move(member_index)), m_is_arrow(is_arrow),
        m_is_synthetic(is_synthetic), m_is_dynamic(is_dynamic),
        m_field_name(name), m_valobj_sp(valobj_sp) {
    if (bitfield_size) {
      m_is_bitfield = true;
      m_bitfield_size = bitfield_size.value();
    } else {
      m_is_bitfield = false;
      m_bitfield_size = 0;
    }
  }

  void Accept(Visitor *v) const override;
  bool is_rvalue() const override { return false; }
  bool is_bitfield() const override { return m_is_bitfield; }
  uint32_t bitfield_size() const override { return m_bitfield_size; }
  CompilerType result_type() const override { return m_result_type; }

  DILASTNode *lhs() const { return m_lhs.get(); }
  const std::vector<uint32_t> &member_index() const { return m_member_index; }
  bool is_arrow() const { return m_is_arrow; }
  bool is_synthetic() const { return m_is_synthetic; }
  bool is_dynamic() const { return m_is_dynamic; }
  ConstString field_name() const { return m_field_name; }
  lldb::ValueObjectSP valobj_sp() const { return m_valobj_sp; }

  static bool classof(const DILASTNode *node) {
    return node->getKind() == NodeKind::eMemberOfNode;
  }

private:
  CompilerType m_result_type;
  DILASTNodeUP m_lhs;
  bool m_is_bitfield;
  uint32_t m_bitfield_size;
  std::vector<uint32_t> m_member_index;
  bool m_is_arrow;
  bool m_is_synthetic;
  bool m_is_dynamic;
  ConstString m_field_name;
  lldb::ValueObjectSP m_valobj_sp;
};

class ArraySubscriptNode : public DILASTNode {
public:
  ArraySubscriptNode(clang::SourceLocation location, CompilerType result_type,
                     DILASTNodeUP base, DILASTNodeUP index)
      : DILASTNode(location, NodeKind::eArraySubscriptNode),
        m_result_type(result_type), m_base(std::move(base)),
        m_index(std::move(index)) {}

  void Accept(Visitor *v) const override;
  bool is_rvalue() const override { return false; }
  CompilerType result_type() const override { return m_result_type; }

  DILASTNode *base() const { return m_base.get(); }
  DILASTNode *index() const { return m_index.get(); }

  static bool classof(const DILASTNode *node) {
    return node->getKind() == NodeKind::eArraySubscriptNode;
  }

private:
  CompilerType m_result_type;
  DILASTNodeUP m_base;
  DILASTNodeUP m_index;
};

class UnaryOpNode : public DILASTNode {
public:
  UnaryOpNode(clang::SourceLocation location, CompilerType result_type,
              UnaryOpKind kind, DILASTNodeUP rhs)
      : DILASTNode(location, NodeKind::eUnaryOpNode),
        m_result_type(result_type), m_kind(kind), m_rhs(std::move(rhs)) {}

  void Accept(Visitor *v) const override;
  bool is_rvalue() const override { return m_kind != UnaryOpKind::Deref; }
  CompilerType result_type() const override { return m_result_type; }

  UnaryOpKind kind() const { return m_kind; }
  DILASTNode *rhs() const { return m_rhs.get(); }

  static bool classof(const DILASTNode *node) {
    return node->getKind() == NodeKind::eUnaryOpNode;
  }

private:
  CompilerType m_result_type;
  UnaryOpKind m_kind;
  DILASTNodeUP m_rhs;
};

class SmartPtrToPtrDecay : public DILASTNode {
public:
  SmartPtrToPtrDecay(clang::SourceLocation location, CompilerType result_type,
                     DILASTNodeUP ptr)
      : DILASTNode(location, NodeKind::eSmartPtrToPtrDecayNode),
        m_result_type(result_type), m_ptr(std::move(ptr)) {}

  void Accept(Visitor *v) const override;
  bool is_rvalue() const override { return false; }
  CompilerType result_type() const override { return m_result_type; }

  DILASTNode *ptr() const { return m_ptr.get(); }

  static bool classof(const DILASTNode *node) {
    return node->getKind() == NodeKind::eSmartPtrToPtrDecayNode;
  }

private:
  CompilerType m_result_type;
  DILASTNodeUP m_ptr;
};

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
  virtual void Visit(const CStyleCastNode *node) = 0;
  virtual void Visit(const MemberOfNode *node) = 0;
  virtual void Visit(const ArraySubscriptNode *node) = 0;
  virtual void Visit(const UnaryOpNode *node) = 0;
  virtual void Visit(const SmartPtrToPtrDecay *node) = 0;
};

} // namespace DIL

} // namespace lldb_private

#endif // LLDB_CORE_DILAST_H
