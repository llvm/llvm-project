//===--- ASTMatchersInternal.cpp - Structural query framework -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Implements the base layer of the matcher framework.
//
//===----------------------------------------------------------------------===//

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersInternal.h"

namespace clang {
namespace ast_matchers {
namespace internal {

void BoundNodesMap::copyTo(BoundNodesTreeBuilder *Builder) const {
  for (IDToNodeMap::const_iterator It = NodeMap.begin();
       It != NodeMap.end();
       ++It) {
    Builder->setBinding(It->first, It->second);
  }
}

void BoundNodesMap::copyTo(BoundNodesMap *Other) const {
  copy(NodeMap.begin(), NodeMap.end(),
       inserter(Other->NodeMap, Other->NodeMap.begin()));
}

BoundNodesTree::BoundNodesTree() {}

BoundNodesTree::BoundNodesTree(
  const BoundNodesMap& Bindings,
  const std::vector<BoundNodesTree> RecursiveBindings)
  : Bindings(Bindings),
    RecursiveBindings(RecursiveBindings) {}

void BoundNodesTree::copyTo(BoundNodesTreeBuilder* Builder) const {
  Bindings.copyTo(Builder);
  for (std::vector<BoundNodesTree>::const_iterator
         I = RecursiveBindings.begin(),
         E = RecursiveBindings.end();
       I != E; ++I) {
    Builder->addMatch(*I);
  }
}

void BoundNodesTree::visitMatches(Visitor* ResultVisitor) {
  BoundNodesMap AggregatedBindings;
  visitMatchesRecursively(ResultVisitor, AggregatedBindings);
}

void BoundNodesTree::
visitMatchesRecursively(Visitor* ResultVisitor,
                        const BoundNodesMap& AggregatedBindings) {
  BoundNodesMap CombinedBindings(AggregatedBindings);
  Bindings.copyTo(&CombinedBindings);
  if (RecursiveBindings.empty()) {
    ResultVisitor->visitMatch(BoundNodes(CombinedBindings));
  } else {
    for (unsigned I = 0; I < RecursiveBindings.size(); ++I) {
      RecursiveBindings[I].visitMatchesRecursively(ResultVisitor,
                                                   CombinedBindings);
    }
  }
}

BoundNodesTreeBuilder::BoundNodesTreeBuilder() {}

void BoundNodesTreeBuilder::addMatch(const BoundNodesTree& Bindings) {
  RecursiveBindings.push_back(Bindings);
}

BoundNodesTree BoundNodesTreeBuilder::build() const {
  return BoundNodesTree(Bindings, RecursiveBindings);
}

} // end namespace internal
} // end namespace ast_matchers
} // end namespace clang
