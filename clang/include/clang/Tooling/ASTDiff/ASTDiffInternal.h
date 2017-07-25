//===- ASTDiffInternal.h --------------------------------------*- C++ -*- -===//
//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_ASTDIFF_ASTDIFFINTERNAL_H
#define LLVM_CLANG_TOOLING_ASTDIFF_ASTDIFFINTERNAL_H

#include <utility>

#include "clang/AST/ASTTypeTraits.h"

namespace clang {
namespace diff {

using DynTypedNode = ast_type_traits::DynTypedNode;

struct ComparisonOptions;
class SyntaxTree;

/// Within a tree, this identifies a node by its preorder offset.
struct NodeId {
private:
  static constexpr int InvalidNodeId = -1;

public:
  int Id;

  NodeId() : Id(InvalidNodeId) {}
  NodeId(int Id) : Id(Id) {}

  operator int() const { return Id; }
  NodeId &operator++() { return ++Id, *this; }
  NodeId &operator--() { return --Id, *this; }

  bool isValid() const { return Id != InvalidNodeId; }
  bool isInvalid() const { return Id == InvalidNodeId; }
};

/// This represents a match between two nodes in the source and destination
/// trees, meaning that they are likely to be related.
struct Match {
  NodeId Src, Dst;
};

enum ChangeKind {
  Delete, // (Src): delete node Src.
  Update, // (Src, Dst): update the value of node Src to match Dst.
  Insert, // (Src, Dst, Pos): insert Src as child of Dst at offset Pos.
  Move    // (Src, Dst, Pos): move Src to be a child of Dst at offset Pos.
};

struct Change {
  ChangeKind Kind;
  NodeId Src, Dst;
  size_t Position;

  Change(ChangeKind Kind, NodeId Src, NodeId Dst, size_t Position)
      : Kind(Kind), Src(Src), Dst(Dst), Position(Position) {}
  Change(ChangeKind Kind, NodeId Src) : Kind(Kind), Src(Src) {}
  Change(ChangeKind Kind, NodeId Src, NodeId Dst)
      : Kind(Kind), Src(Src), Dst(Dst) {}
};

/// Represents a Clang AST node, alongside some additional information.
struct Node {
  NodeId Parent, LeftMostDescendant, RightMostDescendant;
  int Depth, Height;
  DynTypedNode ASTNode;
  SmallVector<NodeId, 4> Children;

  ast_type_traits::ASTNodeKind getType() const { return ASTNode.getNodeKind(); }
  const StringRef getTypeLabel() const { return getType().asStringRef(); }
  bool isLeaf() const { return Children.empty(); }
};

/// Maps nodes of the left tree to ones on the right, and vice versa.
class Mapping {
public:
  Mapping() = default;
  Mapping(Mapping &&Other) = default;
  Mapping &operator=(Mapping &&Other) = default;
  Mapping(int Size1, int Size2) {
    // Maximum possible size after patching one tree.
    int Size = Size1 + Size2;
    SrcToDst = llvm::make_unique<SmallVector<NodeId, 2>[]>(Size);
    DstToSrc = llvm::make_unique<SmallVector<NodeId, 2>[]>(Size);
  }

  void link(NodeId Src, NodeId Dst) {
    SrcToDst[Src].push_back(Dst);
    DstToSrc[Dst].push_back(Src);
  }

  NodeId getDst(NodeId Src) const {
    if (hasSrc(Src))
      return SrcToDst[Src][0];
    return NodeId();
  }
  NodeId getSrc(NodeId Dst) const {
    if (hasDst(Dst))
      return DstToSrc[Dst][0];
    return NodeId();
  }
  const SmallVector<NodeId, 2> &getAllDsts(NodeId Src) const {
    return SrcToDst[Src];
  }
  const SmallVector<NodeId, 2> &getAllSrcs(NodeId Dst) const {
    return DstToSrc[Dst];
  }
  bool hasSrc(NodeId Src) const { return !SrcToDst[Src].empty(); }
  bool hasDst(NodeId Dst) const { return !DstToSrc[Dst].empty(); }
  bool hasSrcDst(NodeId Src, NodeId Dst) const {
    for (NodeId DstId : SrcToDst[Src])
      if (DstId == Dst)
        return true;
    for (NodeId SrcId : DstToSrc[Dst])
      if (SrcId == Src)
        return true;
    return false;
  }

private:
  std::unique_ptr<SmallVector<NodeId, 2>[]> SrcToDst, DstToSrc;
};

/// Represents the AST of a TranslationUnit.
class SyntaxTreeImpl {
public:
  /// Constructs a tree from the entire translation unit.
  SyntaxTreeImpl(SyntaxTree *Parent, const ASTContext &AST);
  /// Constructs a tree from an AST node.
  SyntaxTreeImpl(SyntaxTree *Parent, Decl *N, const ASTContext &AST);
  SyntaxTreeImpl(SyntaxTree *Parent, Stmt *N, const ASTContext &AST);
  template <class T>
  SyntaxTreeImpl(
      SyntaxTree *Parent,
      typename std::enable_if<std::is_base_of<Stmt, T>::value, T>::type *Node,
      const ASTContext &AST)
      : SyntaxTreeImpl(Parent, dyn_cast<Stmt>(Node), AST) {}
  template <class T>
  SyntaxTreeImpl(
      SyntaxTree *Parent,
      typename std::enable_if<std::is_base_of<Decl, T>::value, T>::type *Node,
      const ASTContext &AST)
      : SyntaxTreeImpl(Parent, dyn_cast<Decl>(Node), AST) {}

  SyntaxTree *Parent;
  const ASTContext &AST;
  std::vector<NodeId> Leaves;
  // Maps preorder indices to postorder ones.
  std::vector<int> PostorderIds;

  int getSize() const { return Nodes.size(); }
  NodeId root() const { return 0; }

  const Node &getNode(NodeId Id) const { return Nodes[Id]; }
  Node &getMutableNode(NodeId Id) { return Nodes[Id]; }
  bool isValidNodeId(NodeId Id) const { return Id >= 0 && Id < getSize(); }
  void addNode(Node &N) { Nodes.push_back(N); }
  int getNumberOfDescendants(NodeId Id) const;
  bool isInSubtree(NodeId Id, NodeId SubtreeRoot) const;

  std::string getNodeValueImpl(NodeId Id) const;
  std::string getNodeValueImpl(const DynTypedNode &DTN) const;
  /// Prints the node as "<type>[: <value>](<postorder-id)"
  void printNode(NodeId Id) const { printNode(llvm::outs(), Id); }
  void printNode(raw_ostream &OS, NodeId Id) const;

  void printTree() const;
  void printTree(NodeId Root) const;
  void printTree(raw_ostream &OS, NodeId Root) const;

  void printAsJsonImpl(raw_ostream &OS) const;
  void printNodeAsJson(raw_ostream &OS, NodeId Id) const;

private:
  /// Nodes in preorder.
  std::vector<Node> Nodes;

  void initTree();
  void setLeftMostDescendants();
};

} // end namespace diff
} // end namespace clang
#endif
