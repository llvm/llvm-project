//===- unittest/AST/DeclMatcher.h - AST unit test support ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_UNITTESTS_AST_DECLMATCHER_H
#define LLVM_CLANG_UNITTESTS_AST_DECLMATCHER_H

#include "clang/ASTMatchers/ASTMatchFinder.h"

namespace clang {
namespace ast_matchers {

enum class DeclMatcherKind { First, Last };

// Matcher class to retrieve the first/last matched node under a given AST.
template <typename NodeType, DeclMatcherKind MatcherKind>
class DeclMatcher : public MatchFinder::MatchCallback {
  NodeType *Node = nullptr;
  void run(const MatchFinder::MatchResult &Result) override {
    if ((MatcherKind == DeclMatcherKind::First && Node == nullptr) ||
        MatcherKind == DeclMatcherKind::Last) {
      Node = const_cast<NodeType *>(Result.Nodes.getNodeAs<NodeType>(""));
    }
  }
public:
  // Returns the first/last matched node under the tree rooted in `D`.
  template <typename MatcherType>
  NodeType *match(const Decl *D, const MatcherType &AMatcher) {
    MatchFinder Finder;
    Finder.addMatcher(AMatcher.bind(""), this);
    Finder.matchAST(D->getASTContext());
    assert(Node);
    return Node;
  }
};
template <typename NodeType>
using LastDeclMatcher = DeclMatcher<NodeType, DeclMatcherKind::Last>;
template <typename NodeType>
using FirstDeclMatcher = DeclMatcher<NodeType, DeclMatcherKind::First>;

template <typename NodeType>
class DeclCounterWithPredicate : public MatchFinder::MatchCallback {
  using UnaryPredicate = std::function<bool(const NodeType *)>;
  UnaryPredicate Predicate;
  unsigned Count = 0;
  void run(const MatchFinder::MatchResult &Result) override {
    if (auto N = Result.Nodes.getNodeAs<NodeType>("")) {
      if (Predicate(N))
        ++Count;
    }
  }

public:
  DeclCounterWithPredicate()
      : Predicate([](const NodeType *) { return true; }) {}
  DeclCounterWithPredicate(UnaryPredicate P) : Predicate(P) {}
  // Returns the number of matched nodes which satisfy the predicate under the
  // tree rooted in `D`.
  template <typename MatcherType>
  unsigned match(const Decl *D, const MatcherType &AMatcher) {
    MatchFinder Finder;
    Finder.addMatcher(AMatcher.bind(""), this);
    Finder.matchAST(D->getASTContext());
    return Count;
  }
};

template <typename NodeType>
using DeclCounter = DeclCounterWithPredicate<NodeType>;

} // end namespace ast_matchers
} // end namespace clang

#endif
