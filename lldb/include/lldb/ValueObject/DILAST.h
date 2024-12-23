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
#include <string>
#include <vector>

#include "lldb/ValueObject/ValueObject.h"

namespace lldb_private {

namespace dil {

/// The various types DIL AST nodes (used by the DIL parser).
enum class NodeKind {
  eErrorNode,
  eIdentifierNode,
};

/// Class used to store & manipulate information about identifiers.
class IdentifierInfo {
public:
  enum class Kind {
    eValue,
    eContextArg,
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

  Kind GetKind() const { return m_kind; }
  lldb::ValueObjectSP GetValue() const { return m_value; }

  CompilerType GetType() { return m_type; }
  bool IsValid() const { return m_type.IsValid(); }

  IdentifierInfo(Kind kind, CompilerType type, lldb::ValueObjectSP value,
                 std::vector<uint32_t> path)
      : m_kind(kind), m_type(type), m_value(std::move(value)) {}

private:
  Kind m_kind;
  CompilerType m_type;
  lldb::ValueObjectSP m_value;
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
/// expression evaluator. The DIL parser parses the input string and creates
/// the AST parse tree from the AST nodes. The resulting AST node tree gets
/// passed to the DIL expression evaluator, which evaluates the DIL AST nodes
/// and creates/returns a ValueObjectSP containing the result.

/// Base class for AST nodes used by the Data Inspection Language (DIL) parser.
/// All of the specialized types of AST nodes inherit from this (virtual) base
/// class.
class DILASTNode {
public:
  DILASTNode(uint32_t location, NodeKind kind)
      : m_location(location), m_kind(kind) {}
  virtual ~DILASTNode() = default;

  virtual void Accept(Visitor *v) const = 0;

  uint32_t GetLocation() const { return m_location; }
  NodeKind GetKind() const { return m_kind; }

private:
  uint32_t m_location;
  const NodeKind m_kind;
};

using DILASTNodeUP = std::unique_ptr<DILASTNode>;

class ErrorNode : public DILASTNode {
public:
  ErrorNode(CompilerType empty_type)
      : DILASTNode(0, NodeKind::eErrorNode), m_empty_type(empty_type) {}
  void Accept(Visitor *v) const override;

  static bool classof(const DILASTNode *node) {
    return node->GetKind() == NodeKind::eErrorNode;
  }

private:
  CompilerType m_empty_type;
};

class IdentifierNode : public DILASTNode {
public:
  IdentifierNode(uint32_t location, std::string name,
                 lldb::DynamicValueType use_dynamic,
                 std::shared_ptr<ExecutionContextScope> exe_ctx_scope)
      : DILASTNode(location, NodeKind::eIdentifierNode),
        m_name(std::move(name)), m_use_dynamic(use_dynamic),
        m_ctx_scope(exe_ctx_scope) {}

  void Accept(Visitor *v) const override;

  lldb::DynamicValueType use_dynamic() const { return m_use_dynamic; }
  std::string name() const { return m_name; }
  std::shared_ptr<ExecutionContextScope> get_exe_context() const {
    return m_ctx_scope;
  }

  static bool classof(const DILASTNode *node) {
    return node->GetKind() == NodeKind::eIdentifierNode;
  }

private:
  std::string m_name;
  lldb::DynamicValueType m_use_dynamic;
  std::shared_ptr<ExecutionContextScope> m_ctx_scope;
};

/// This class contains one Visit method for each specialized type of
/// DIL AST node. The Visit methods are used to dispatch a DIL AST node to
/// the correct function in the DIL expression evaluator for evaluating that
/// type of AST node.
class Visitor {
public:
  virtual ~Visitor() = default;
  virtual void Visit(const ErrorNode *node) = 0;
  virtual void Visit(const IdentifierNode *node) = 0;
};

} // namespace dil

} // namespace lldb_private

#endif // LLDB_VALUEOBJECT_DILAST_H
