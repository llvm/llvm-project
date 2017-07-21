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

class SyntaxTree;

class ASTDiff {
public:
  ASTDiff(SyntaxTree &T1, SyntaxTree &T2, const ComparisonOptions &Options);
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
      : TreeImpl(llvm::make_unique<SyntaxTreeImpl>(this, Node, AST)) {}

  /// Serialize the node attributes to a string representation. This should
  /// uniquely distinguish nodes of the same kind. Note that this function just
  /// returns a representation of the node value, not considering descendants.
  std::string getNodeValue(const DynTypedNode &DTN) const;

  void printAsJson(raw_ostream &OS);

  std::unique_ptr<SyntaxTreeImpl> TreeImpl;
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

  /// Returns zero if the nodes are considered to be equal.  Returns a value
  /// indicating the editing distance between the nodes otherwise.
  /// There is no need to consider nodes that cannot be matched as input for
  /// this function (see isMatchingAllowed).
  double getNodeDistance(const SyntaxTree &T1, const DynTypedNode &N1,
                         const SyntaxTree &T2, const DynTypedNode &N2) const {
    if (T1.getNodeValue(N1) == T2.getNodeValue(N2))
      return 0;
    return 1;
  }
};

} // end namespace diff
} // end namespace clang

#endif
