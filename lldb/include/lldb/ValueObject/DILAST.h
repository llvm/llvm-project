//===-- DILAST.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_DIL_AST_H_
#define LLDB_DIL_AST_H_

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

namespace lldb_private {

/// Struct to hold information about member fields. Used by the parser for the
/// Data Inspection Language (DIL).
struct DILMemberInfo {
  std::optional<std::string> name;
  CompilerType type;
  bool is_bitfield;
  uint32_t bitfield_size_in_bits;
  bool is_synthetic;
  bool is_dynamic;
  lldb::ValueObjectSP val_obj_sp;

  explicit operator bool() const { return type.IsValid(); }
};

/// This determines if the type is a shared, unique or weak pointer, either
/// from stdlibc++ or libc+++.
bool IsSmartPtrType(CompilerType type);

/// Finds the member field with the given name and type, stores the child index
/// corresponding to the field in the idx vector and returns a DILMemberInfo
/// struct with appropriate information about the field.
DILMemberInfo GetFieldWithNameIndexPath(lldb::ValueObjectSP lhs_val_sp,
                                        CompilerType type,
                                        const std::string &name,
                                        std::vector<uint32_t> *idx,
                                        CompilerType empty_type,
                                        bool use_synthetic, bool is_dynamic);

std::tuple<DILMemberInfo, std::vector<uint32_t>>
GetMemberInfo(lldb::ValueObjectSP lhs_val_sp, CompilerType type,
              const std::string &name, bool use_synthetic);

/// Get the appropriate ValueObjectSP, consulting the use_dynamic and
/// use_synthetic options passed, acquiring the process & target locks if
/// appropriate.
lldb::ValueObjectSP
DILGetSPWithLock(lldb::ValueObjectSP valobj_sp,
                 lldb::DynamicValueType use_dynamic = lldb::eNoDynamicValues,
                 bool use_synthetic = false);

/// The various types DIL AST nodes (used by the DIL parser).
enum class DILNodeKind {
  kDILErrorNode,
  kLiteralNode,
  kIdentifierNode,
  kSizeOfNode,
  kBuiltinFunctionCallNode,
  kCStyleCastNode,
  kCxxStaticCastNode,
  kCxxReinterpretCastNode,
  kMemberOfNode,
  kArraySubscriptNode,
  kBinaryOpNode,
  kUnaryOpNode,
  kTernaryOpNode,
  kSmartPtrToPtrDecay
};

/// The C-Style casts allowed by DIL.
enum class CStyleCastKind {
  kArithmetic,
  kEnumeration,
  kPointer,
  kNullptr,
  kReference,
};

/// The Cxx static casts allowed by DIL.
enum class CxxStaticCastKind {
  kNoOp,
  kArithmetic,
  kEnumeration,
  kPointer,
  kNullptr,
  kBaseToDerived,
  kDerivedToBase,
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

/// Quick lookup to check if a type name already exists in a
/// name-to-CompilerType map the DIL parser keeps of previously found
/// name/type pairs.
bool IsContextVar(const std::string &name);

/// Checks to see if the CompilerType is a Smart Pointer (shared, unique, weak)
/// or not. Only applicable for C++, which is why this is here and not part of
/// the CompilerType class.
bool IsSmartPtrType(CompilerType type);

/// Class used to store & manipulate information about identifiers.
class IdentifierInfo {
private:
  using MemberPath = std::vector<uint32_t>;
  using IdentifierInfoPtr = std::unique_ptr<IdentifierInfo>;

public:
  enum class Kind {
    kValue,
    kContextArg,
    kMemberPath,
    kThisKeyword,
  };

  static IdentifierInfoPtr FromValue(lldb::ValueObjectSP value_sp) {
    CompilerType type;
    lldb::ValueObjectSP value = DILGetSPWithLock(value_sp);
    if (value)
      type = value->GetCompilerType();
    return IdentifierInfoPtr(new IdentifierInfo(Kind::kValue, type, value, {}));
  }

  static IdentifierInfoPtr FromContextArg(CompilerType type) {
    lldb::ValueObjectSP empty_value;
    return IdentifierInfoPtr(
        new IdentifierInfo(Kind::kContextArg, type, empty_value, {}));
  }

  static IdentifierInfoPtr FromMemberPath(CompilerType type, MemberPath path) {
    lldb::ValueObjectSP empty_value;
    return IdentifierInfoPtr(new IdentifierInfo(Kind::kMemberPath, type,
                                                empty_value, std::move(path)));
  }

  static IdentifierInfoPtr FromThisKeyword(CompilerType type) {
    lldb::ValueObjectSP empty_value;
    return IdentifierInfoPtr(
        new IdentifierInfo(Kind::kThisKeyword, type, empty_value, {}));
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
class DILVisitor;

/// The rest of the classes in this file, except for the DILVisitor class at the
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
  DILASTNode(clang::SourceLocation location) : location_(location) {}
  virtual ~DILASTNode() {}

  virtual void Accept(DILVisitor *v) const = 0;

  virtual bool is_error() const { return false; };
  virtual bool is_rvalue() const = 0;
  virtual bool is_bitfield() const { return false; };
  virtual bool is_context_var() const { return false; };
  virtual bool is_literal_zero() const { return false; }
  virtual uint32_t bitfield_size() const { return 0; }
  virtual CompilerType result_type() const = 0;

  virtual DILNodeKind what_am_i() const = 0;

  clang::SourceLocation location() const { return location_; }

  // The expression result type, but dereferenced in case it's a reference. This
  // is for convenience, since for the purposes of the semantic analysis only
  // the dereferenced type matters.
  CompilerType result_type_deref() const;

private:
  clang::SourceLocation location_;
};

using ParseResult = std::unique_ptr<DILASTNode>;

class DILErrorNode : public DILASTNode {
public:
  DILErrorNode(CompilerType empty_type)
      : DILASTNode(clang::SourceLocation()), m_empty_type(empty_type) {}
  void Accept(DILVisitor *v) const override;
  bool is_error() const override { return true; }
  bool is_rvalue() const override { return false; }
  CompilerType result_type() const override { return m_empty_type; }
  CompilerType result_type_real() const { return m_empty_type; }
  DILNodeKind what_am_i() const override { return DILNodeKind::kDILErrorNode; }

private:
  CompilerType m_empty_type;
};

class LiteralNode : public DILASTNode {
public:
  template <typename ValueT>
  LiteralNode(clang::SourceLocation location, CompilerType type, ValueT &&value,
              bool is_literal_zero)
      : DILASTNode(location), m_type(type),
        m_value(std::forward<ValueT>(value)),
        m_is_literal_zero(is_literal_zero) {}

  void Accept(DILVisitor *v) const override;
  bool is_rvalue() const override { return true; }
  bool is_literal_zero() const override { return m_is_literal_zero; }
  CompilerType result_type() const override { return m_type; }
  DILNodeKind what_am_i() const override { return DILNodeKind::kLiteralNode; }

  template <typename ValueT> ValueT value() const {
    return std::get<ValueT>(m_value);
  }

  auto value() const { return m_value; }

private:
  CompilerType m_type;
  std::variant<llvm::APInt, llvm::APFloat, bool, std::vector<char>> m_value;
  bool m_is_literal_zero;
};

class IdentifierNode : public DILASTNode {
public:
  IdentifierNode(clang::SourceLocation location, std::string name,
                 std::unique_ptr<IdentifierInfo> identifier, bool is_rvalue,
                 bool is_context_var)
      : DILASTNode(location), m_is_rvalue(is_rvalue),
        m_is_context_var(is_context_var), m_name(std::move(name)),
        m_identifier(std::move(identifier)) {}

  void Accept(DILVisitor *v) const override;
  bool is_rvalue() const override { return m_is_rvalue; }
  bool is_context_var() const override { return m_is_context_var; };
  CompilerType result_type() const override { return m_identifier->GetType(); }
  DILNodeKind what_am_i() const override {
    return DILNodeKind::kIdentifierNode;
  }

  std::string name() const { return m_name; }
  const IdentifierInfo &info() const { return *m_identifier; }

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
      : DILASTNode(location), m_type(type), m_operand(operand) {}

  void Accept(DILVisitor *v) const override;
  bool is_rvalue() const override { return true; }
  CompilerType result_type() const override { return m_type; }
  DILNodeKind what_am_i() const override { return DILNodeKind::kSizeOfNode; }

  CompilerType operand() const { return m_operand; }

private:
  CompilerType m_type;
  CompilerType m_operand;
};

class BuiltinFunctionCallNode : public DILASTNode {
public:
  BuiltinFunctionCallNode(clang::SourceLocation location,
                          CompilerType result_type, std::string name,
                          std::vector<ParseResult> arguments)
      : DILASTNode(location), m_result_type(result_type),
        m_name(std::move(name)), m_arguments(std::move(arguments)) {}

  void Accept(DILVisitor *v) const override;
  bool is_rvalue() const override { return true; }
  CompilerType result_type() const override { return m_result_type; }
  DILNodeKind what_am_i() const override {
    return DILNodeKind::kBuiltinFunctionCallNode;
  }

  std::string name() const { return m_name; }
  const std::vector<ParseResult> &arguments() const { return m_arguments; };

private:
  CompilerType m_result_type;
  std::string m_name;
  std::vector<ParseResult> m_arguments;
};

class CStyleCastNode : public DILASTNode {
public:
  CStyleCastNode(clang::SourceLocation location, CompilerType type,
                 ParseResult rhs, CStyleCastKind kind)
      : DILASTNode(location), m_type(type), m_rhs(std::move(rhs)),
        m_kind(kind) {}

  void Accept(DILVisitor *v) const override;
  bool is_rvalue() const override {
    return m_kind != CStyleCastKind::kReference;
  }
  CompilerType result_type() const override { return m_type; }
  DILNodeKind what_am_i() const override {
    return DILNodeKind::kCStyleCastNode;
  }

  CompilerType type() const { return m_type; }
  DILASTNode *rhs() const { return m_rhs.get(); }
  CStyleCastKind kind() const { return m_kind; }

private:
  CompilerType m_type;
  ParseResult m_rhs;
  CStyleCastKind m_kind;
};

class CxxStaticCastNode : public DILASTNode {
public:
  CxxStaticCastNode(clang::SourceLocation location, CompilerType type,
                    ParseResult rhs, CxxStaticCastKind kind, bool is_rvalue)
      : DILASTNode(location), m_type(type), m_rhs(std::move(rhs)), m_kind(kind),
        m_is_rvalue(is_rvalue) {
    assert(kind != CxxStaticCastKind::kBaseToDerived &&
           kind != CxxStaticCastKind::kDerivedToBase &&
           "invalid constructor for base-to-derived and derived-to-base casts");
  }

  CxxStaticCastNode(clang::SourceLocation location, CompilerType type,
                    ParseResult rhs, std::vector<uint32_t> idx, bool is_rvalue)
      : DILASTNode(location), m_type(type), m_rhs(std::move(rhs)),
        m_idx(std::move(idx)), m_kind(CxxStaticCastKind::kDerivedToBase),
        m_is_rvalue(is_rvalue) {}

  CxxStaticCastNode(clang::SourceLocation location, CompilerType type,
                    ParseResult rhs, uint64_t offset, bool is_rvalue)
      : DILASTNode(location), m_type(type), m_rhs(std::move(rhs)),
        m_offset(offset), m_kind(CxxStaticCastKind::kBaseToDerived),
        m_is_rvalue(is_rvalue) {}

  void Accept(DILVisitor *v) const override;
  bool is_rvalue() const override { return m_is_rvalue; }
  CompilerType result_type() const override { return m_type; }
  DILNodeKind what_am_i() const override {
    return DILNodeKind::kCxxStaticCastNode;
  }

  CompilerType type() const { return m_type; }
  DILASTNode *rhs() const { return m_rhs.get(); }
  const std::vector<uint32_t> &idx() const { return m_idx; }
  uint64_t offset() const { return m_offset; }
  CxxStaticCastKind kind() const { return m_kind; }

private:
  CompilerType m_type;
  ParseResult m_rhs;
  std::vector<uint32_t> m_idx;
  uint64_t m_offset = 0;
  CxxStaticCastKind m_kind;
  bool m_is_rvalue;
};

class CxxReinterpretCastNode : public DILASTNode {
public:
  CxxReinterpretCastNode(clang::SourceLocation location, CompilerType type,
                         ParseResult rhs, bool is_rvalue)
      : DILASTNode(location), m_type(type), m_rhs(std::move(rhs)),
        m_is_rvalue(is_rvalue) {}

  void Accept(DILVisitor *v) const override;
  bool is_rvalue() const override { return m_is_rvalue; }
  CompilerType result_type() const override { return m_type; }
  DILNodeKind what_am_i() const override {
    return DILNodeKind::kCxxReinterpretCastNode;
  }

  CompilerType type() const { return m_type; }
  DILASTNode *rhs() const { return m_rhs.get(); }

private:
  CompilerType m_type;
  ParseResult m_rhs;
  bool m_is_rvalue;
};

class MemberOfNode : public DILASTNode {
public:
  MemberOfNode(clang::SourceLocation location, CompilerType result_type,
               ParseResult lhs, bool is_bitfield, uint32_t bitfield_size,
               std::vector<uint32_t> member_index, bool is_arrow,
               bool is_synthetic, bool is_dynamic, ConstString name,
               lldb::ValueObjectSP valobj_sp)
      : DILASTNode(location), m_result_type(result_type), m_lhs(std::move(lhs)),
        m_is_bitfield(is_bitfield), m_bitfield_size(bitfield_size),
        m_member_index(std::move(member_index)), m_is_arrow(is_arrow),
        m_is_synthetic(is_synthetic), m_is_dynamic(is_dynamic),
        m_field_name(name), m_valobj_sp(valobj_sp) {}

  void Accept(DILVisitor *v) const override;
  bool is_rvalue() const override { return false; }
  bool is_bitfield() const override { return m_is_bitfield; }
  uint32_t bitfield_size() const override { return m_bitfield_size; }
  CompilerType result_type() const override { return m_result_type; }
  DILNodeKind what_am_i() const override { return DILNodeKind::kMemberOfNode; }

  DILASTNode *lhs() const { return m_lhs.get(); }
  const std::vector<uint32_t> &member_index() const { return m_member_index; }
  bool is_arrow() const { return m_is_arrow; }
  bool is_synthetic() const { return m_is_synthetic; }
  bool is_dynamic() const { return m_is_dynamic; }
  ConstString field_name() const { return m_field_name; }
  lldb::ValueObjectSP valobj_sp() const { return m_valobj_sp; }

private:
  CompilerType m_result_type;
  ParseResult m_lhs;
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
                     ParseResult base, ParseResult index)
      : DILASTNode(location), m_result_type(result_type),
        m_base(std::move(base)), m_index(std::move(index)) {}

  void Accept(DILVisitor *v) const override;
  bool is_rvalue() const override { return false; }
  CompilerType result_type() const override { return m_result_type; }
  DILNodeKind what_am_i() const override {
    return DILNodeKind::kArraySubscriptNode;
  }

  DILASTNode *base() const { return m_base.get(); }
  DILASTNode *index() const { return m_index.get(); }

private:
  CompilerType m_result_type;
  ParseResult m_base;
  ParseResult m_index;
};

class BinaryOpNode : public DILASTNode {
public:
  BinaryOpNode(clang::SourceLocation location, CompilerType result_type,
               BinaryOpKind kind, ParseResult lhs, ParseResult rhs,
               CompilerType comp_assign_type,
               ValueObject *val_obj_ptr = nullptr)
      : DILASTNode(location), m_result_type(result_type), m_kind(kind),
        m_lhs(std::move(lhs)), m_rhs(std::move(rhs)),
        m_comp_assign_type(comp_assign_type) {
    if (val_obj_ptr)
      m_val_obj_sp = val_obj_ptr->GetSP();
  }

  void Accept(DILVisitor *v) const override;
  bool is_rvalue() const override {
    return !binary_op_kind_is_comp_assign(m_kind);
  }
  CompilerType result_type() const override { return m_result_type; }
  DILNodeKind what_am_i() const override { return DILNodeKind::kBinaryOpNode; }

  BinaryOpKind kind() const { return m_kind; }
  DILASTNode *lhs() const { return m_lhs.get(); }
  DILASTNode *rhs() const { return m_rhs.get(); }
  CompilerType comp_assign_type() const { return m_comp_assign_type; }
  lldb::ValueObjectSP get_valobj_sp() const { return m_val_obj_sp; }

private:
  CompilerType m_result_type;
  BinaryOpKind m_kind;
  ParseResult m_lhs;
  ParseResult m_rhs;
  CompilerType m_comp_assign_type;
  lldb::ValueObjectSP m_val_obj_sp;
};

class UnaryOpNode : public DILASTNode {
public:
  UnaryOpNode(clang::SourceLocation location, CompilerType result_type,
              UnaryOpKind kind, ParseResult rhs)
      : DILASTNode(location), m_result_type(result_type), m_kind(kind),
        m_rhs(std::move(rhs)) {}

  void Accept(DILVisitor *v) const override;
  bool is_rvalue() const override { return m_kind != UnaryOpKind::Deref; }
  CompilerType result_type() const override { return m_result_type; }
  DILNodeKind what_am_i() const override { return DILNodeKind::kUnaryOpNode; }

  UnaryOpKind kind() const { return m_kind; }
  DILASTNode *rhs() const { return m_rhs.get(); }

private:
  CompilerType m_result_type;
  UnaryOpKind m_kind;
  ParseResult m_rhs;
};

class TernaryOpNode : public DILASTNode {
public:
  TernaryOpNode(clang::SourceLocation location, CompilerType result_type,
                ParseResult cond, ParseResult lhs, ParseResult rhs)
      : DILASTNode(location), m_result_type(result_type),
        m_cond(std::move(cond)), m_lhs(std::move(lhs)), m_rhs(std::move(rhs)) {}

  void Accept(DILVisitor *v) const override;
  bool is_rvalue() const override {
    return m_lhs->is_rvalue() || m_rhs->is_rvalue();
  }
  bool is_bitfield() const override {
    return m_lhs->is_bitfield() || m_rhs->is_bitfield();
  }
  CompilerType result_type() const override { return m_result_type; }
  DILNodeKind what_am_i() const override { return DILNodeKind::kTernaryOpNode; }

  DILASTNode *cond() const { return m_cond.get(); }
  DILASTNode *lhs() const { return m_lhs.get(); }
  DILASTNode *rhs() const { return m_rhs.get(); }

private:
  CompilerType m_result_type;
  ParseResult m_cond;
  ParseResult m_lhs;
  ParseResult m_rhs;
};

class SmartPtrToPtrDecay : public DILASTNode {
public:
  SmartPtrToPtrDecay(clang::SourceLocation location, CompilerType result_type,
                     ParseResult ptr)
      : DILASTNode(location), m_result_type(result_type),
        m_ptr(std::move(ptr)) {}

  void Accept(DILVisitor *v) const override;
  bool is_rvalue() const override { return false; }
  CompilerType result_type() const override { return m_result_type; }
  DILNodeKind what_am_i() const override {
    return DILNodeKind::kSmartPtrToPtrDecay;
  }

  DILASTNode *ptr() const { return m_ptr.get(); }

private:
  CompilerType m_result_type;
  ParseResult m_ptr;
};

/// This class contains one Visit method for each specialized type of
/// DIL AST node. The Visit methods are used to dispatch a DIL AST node to
/// the correct function in the DIL expression evaluator for evaluating that
/// type of AST node.
class DILVisitor {
public:
  virtual ~DILVisitor() {}
  virtual void Visit(const DILErrorNode *node) = 0;
  virtual void Visit(const LiteralNode *node) = 0;
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
  virtual void Visit(const SmartPtrToPtrDecay *node) = 0;
};

} // namespace lldb_private

#endif // LLDB_DIL_AST_H_
