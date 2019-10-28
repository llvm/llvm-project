//=== ASTTableGen.h - Common definitions for AST node tablegen --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_AST_TABLEGEN_H
#define CLANG_AST_TABLEGEN_H

#include "llvm/TableGen/Record.h"
#include "llvm/ADT/STLExtras.h"

// These are spellings in the tblgen files.

// Field names that are fortunately common across the hierarchies.
#define BaseFieldName "Base"
#define AbstractFieldName "Abstract"

// Comment node hierarchy.
#define CommentNodeClassName "CommentNode"

// Decl node hierarchy.
#define DeclNodeClassName "DeclNode"
#define DeclContextNodeClassName "DeclContext"

// Stmt node hierarchy.
#define StmtNodeClassName "StmtNode"

// Type node hierarchy.
#define TypeNodeClassName "TypeNode"
#define AlwaysDependentClassName "AlwaysDependent"
#define NeverCanonicalClassName "NeverCanonical"
#define NeverCanonicalUnlessDependentClassName "NeverCanonicalUnlessDependent"
#define LeafTypeClassName "LeafType"

// Property node hierarchy.
#define PropertyClassName "Property"
#define ClassFieldName "Class"

namespace clang {
namespace tblgen {

/// An (optional) reference to a TableGen node representing a class
/// in one of Clang's AST hierarchies.
class ASTNode {
  llvm::Record *Record;
public:
  ASTNode(llvm::Record *record = nullptr) : Record(record) {}

  explicit operator bool() const { return Record != nullptr; }

  llvm::Record *getRecord() const { return Record; }
  llvm::StringRef getName() const {
    assert(Record && "getting name of null record");
    return Record->getName();
  }
  llvm::ArrayRef<llvm::SMLoc> getLoc() const {
    assert(Record && "getting location of null record");
    return Record->getLoc();
  }

  /// Return the node for the base, if there is one.
  ASTNode getBase() const {
    assert(Record && "getting base of null record");
    return Record->getValueAsOptionalDef(BaseFieldName);
  }

  /// Is the corresponding class abstract?
  bool isAbstract() const {
    assert(Record && "querying null record");
    return Record->getValueAsBit(AbstractFieldName);
  }

  /// Does the node inherit from the given TableGen class?
  bool isSubClassOf(llvm::StringRef className) const {
    assert(Record && "querying null record");
    return Record->isSubClassOf(className);
  }

  friend bool operator<(ASTNode lhs, ASTNode rhs) {
    assert(lhs && rhs && "sorting null nodes");
    return lhs.getName() < rhs.getName();
  }
  friend bool operator>(ASTNode lhs, ASTNode rhs) { return rhs < lhs; }
  friend bool operator<=(ASTNode lhs, ASTNode rhs) { return !(rhs < lhs); }
  friend bool operator>=(ASTNode lhs, ASTNode rhs) { return !(lhs < rhs); }

  friend bool operator==(ASTNode lhs, ASTNode rhs) {
    // This should handle null nodes.
    return lhs.getRecord() == rhs.getRecord();
  }
  friend bool operator!=(ASTNode lhs, ASTNode rhs) { return !(lhs == rhs); }
};

class DeclNode : public ASTNode {
public:
  DeclNode(llvm::Record *record = nullptr) : ASTNode(record) {}

  llvm::StringRef getId() const;
  std::string getClassName() const;
  DeclNode getBase() const { return DeclNode(ASTNode::getBase().getRecord()); }
};

class TypeNode : public ASTNode {
public:
  TypeNode(llvm::Record *record = nullptr) : ASTNode(record) {}

  llvm::StringRef getId() const;
  llvm::StringRef getClassName() const;
  TypeNode getBase() const { return TypeNode(ASTNode::getBase().getRecord()); }
};

class StmtNode : public ASTNode {
public:
  StmtNode(llvm::Record *record = nullptr) : ASTNode(record) {}

  std::string getId() const;
  llvm::StringRef getClassName() const;
  StmtNode getBase() const { return StmtNode(ASTNode::getBase().getRecord()); }
};

/// A visitor for an AST node hierarchy.  Note that `base` can be null for
/// the root class.
template <class NodeClass>
using ASTNodeHierarchyVisitor =
  llvm::function_ref<void(NodeClass node, NodeClass base)>;

void visitASTNodeHierarchy(llvm::RecordKeeper &records,
                           llvm::StringRef nodeClassName,
                           ASTNodeHierarchyVisitor<ASTNode> visit);

void visitDeclNodeHierarchy(llvm::RecordKeeper &records,
                            ASTNodeHierarchyVisitor<DeclNode> visit);

void visitTypeNodeHierarchy(llvm::RecordKeeper &records,
                            ASTNodeHierarchyVisitor<TypeNode> visit);

void visitStmtNodeHierarchy(llvm::RecordKeeper &records,
                            ASTNodeHierarchyVisitor<StmtNode> visit);

} // end namespace clang::tblgen
} // end namespace clang

#endif
