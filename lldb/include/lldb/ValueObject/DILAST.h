//===-- DILAST.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_VALUEOBJECT_DILAST_H
#define LLDB_VALUEOBJECT_DILAST_H

#include "lldb/ValueObject/ValueObject.h"
#include <string>

namespace lldb_private::dil {

/// The various types DIL AST nodes (used by the DIL parser).
enum class NodeKind {
  eErrorNode,
  eIdentifierNode,
};

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
class ASTNode {
public:
  ASTNode(uint32_t location, NodeKind kind)
      : m_location(location), m_kind(kind) {}
  virtual ~ASTNode() = default;

  virtual llvm::Expected<lldb::ValueObjectSP> Accept(Visitor *v) const = 0;

  uint32_t GetLocation() const { return m_location; }
  NodeKind GetKind() const { return m_kind; }

private:
  uint32_t m_location;
  const NodeKind m_kind;
};

using ASTNodeUP = std::unique_ptr<ASTNode>;

class ErrorNode : public ASTNode {
public:
  ErrorNode() : ASTNode(0, NodeKind::eErrorNode) {}
  llvm::Expected<lldb::ValueObjectSP> Accept(Visitor *v) const override;

  static bool classof(const ASTNode *node) {
    return node->GetKind() == NodeKind::eErrorNode;
  }
};

class IdentifierNode : public ASTNode {
public:
  IdentifierNode(uint32_t location, std::string name,
                 lldb::DynamicValueType use_dynamic,
                 std::shared_ptr<ExecutionContextScope> exe_ctx_scope)
      : ASTNode(location, NodeKind::eIdentifierNode), m_name(std::move(name)),
        m_use_dynamic(use_dynamic) {}

  llvm::Expected<lldb::ValueObjectSP> Accept(Visitor *v) const override;

  lldb::DynamicValueType GetUseDynamic() const { return m_use_dynamic; }
  std::string GetName() const { return m_name; }

  static bool classof(const ASTNode *node) {
    return node->GetKind() == NodeKind::eIdentifierNode;
  }

private:
  std::string m_name;
  lldb::DynamicValueType m_use_dynamic;
};

/// This class contains one Visit method for each specialized type of
/// DIL AST node. The Visit methods are used to dispatch a DIL AST node to
/// the correct function in the DIL expression evaluator for evaluating that
/// type of AST node.
class Visitor {
public:
  virtual ~Visitor() = default;
  virtual llvm::Expected<lldb::ValueObjectSP>
  Visit(const IdentifierNode *node) = 0;
};

} // namespace lldb_private::dil

#endif // LLDB_VALUEOBJECT_DILAST_H
