//===- ASTDiff.h - AST differencing API -----------------------*- C++ -*- -===//
//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file specifies an interface that can be used to compare C++ syntax
// trees.
//
// We use the gumtree algorithm which combines a heuristic top-down search that
// is able to match large subtrees that are equivalent, with an optimal
// algorithm to match small subtrees.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_ASTDIFF_ASTDIFF_H
#define LLVM_CLANG_TOOLING_ASTDIFF_ASTDIFF_H

#include "clang/Tooling/ASTDiff/ASTDiffInternal.h"

namespace clang {
namespace diff {

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
  ast_type_traits::DynTypedNode ASTNode;
  SmallVector<NodeId, 4> Children;

  ast_type_traits::ASTNodeKind getType() const { return ASTNode.getNodeKind(); }
  const StringRef getTypeLabel() const { return getType().asStringRef(); }
  bool isLeaf() const { return Children.empty(); }
};

class ASTDiff {
public:
  ASTDiff(SyntaxTree &Src, SyntaxTree &Dst, const ComparisonOptions &Options);
  ~ASTDiff();

  // Returns a list of matches.
  std::vector<Match> getMatches();
  /// Returns an edit script.
  std::vector<Change> getChanges();

  // Prints an edit action.
  void printChange(raw_ostream &OS, const Change &Chg) const;
  // Prints a match between two nodes.
  void printMatch(raw_ostream &OS, const Match &M) const;

  class Impl;

private:
  std::unique_ptr<Impl> DiffImpl;
};

/// SyntaxTree objects represent subtrees of the AST.
/// They can be constructed from any Decl or Stmt.
class SyntaxTree {
public:
  /// Constructs a tree from a translation unit.
  SyntaxTree(const ASTContext &AST);
  /// Constructs a tree from any AST node.
  template <class T>
  SyntaxTree(T *Node, const ASTContext &AST)
      : TreeImpl(llvm::make_unique<Impl>(this, Node, AST)) {}
  SyntaxTree(SyntaxTree &&Other) = default;
  ~SyntaxTree();

  const Node &getNode(NodeId Id) const;

  /// Serialize the node attributes to a string representation. This should
  /// uniquely distinguish nodes of the same kind. Note that this function just
  /// returns a representation of the node value, not considering descendants.
  std::string getNodeValue(const DynTypedNode &DTN) const;

  void printAsJson(raw_ostream &OS);

  class Impl;
  std::unique_ptr<Impl> TreeImpl;
};

struct ComparisonOptions {
  /// During top-down matching, only consider nodes of at least this height.
  int MinHeight = 2;

  /// During bottom-up matching, match only nodes with at least this value as
  /// the ratio of their common descendants.
  double MinSimilarity = 0.2;

  /// Whenever two subtrees are matched in the bottom-up phase, the optimal
  /// mapping is computed, unless the size of either subtrees exceeds this.
  int MaxSize = 100;

  /// If this is set to true, nodes that have parents that must not be matched
  /// (see NodeComparison) will be allowed to be matched.
  bool EnableMatchingWithUnmatchableParents = false;

  /// Returns false if the nodes should never be matched.
  bool isMatchingAllowed(const DynTypedNode &N1, const DynTypedNode &N2) const {
    return N1.getNodeKind().isSame(N2.getNodeKind());
  }
};

} // end namespace diff
} // end namespace clang

#endif
