//===- TreeTest.cpp ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Syntax/Tree.h"
#include "TreeTestBase.h"
#include "clang/Tooling/Syntax/BuildTree.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace clang::syntax;

namespace {

class TreeTest : public SyntaxTreeTest {
private:
  Tree *createTree(ArrayRef<const Node *> Children) {
    std::vector<std::pair<Node *, NodeRole>> ChildrenWithRoles;
    ChildrenWithRoles.reserve(Children.size());
    for (const auto *Child : Children) {
      ChildrenWithRoles.push_back(
          std::make_pair(deepCopy(*Arena, Child), NodeRole::Unknown));
    }
    return clang::syntax::createTree(*Arena, ChildrenWithRoles,
                                     NodeKind::UnknownExpression);
  }

  // Generate Forests by combining `Children` into `ParentCount` Trees.
  //
  // We do this recursively.
  std::vector<std::vector<const Tree *>>
  generateAllForests(ArrayRef<const Node *> Children, unsigned ParentCount) {
    assert(ParentCount > 0);
    // If there is only one Parent node, then combine `Children` under
    // this Parent.
    if (ParentCount == 1)
      return {{createTree(Children)}};

    // Otherwise, combine `ChildrenCount` children under the last parent and
    // solve the smaller problem without these children and this parent. Do this
    // for every `ChildrenCount` and combine the results.
    std::vector<std::vector<const Tree *>> AllForests;
    for (unsigned ChildrenCount = 0; ChildrenCount <= Children.size();
         ++ChildrenCount) {
      auto *LastParent = createTree(Children.take_back(ChildrenCount));
      for (auto &Forest : generateAllForests(Children.drop_back(ChildrenCount),
                                             ParentCount - 1)) {
        Forest.push_back(LastParent);
        AllForests.push_back(Forest);
      }
    }
    return AllForests;
  }

protected:
  // Generates all trees with a `Base` of `Node`s and `NodeCountPerLayer`
  // `Node`s per layer. An example of Tree with `Base` = {`(`, `)`} and
  // `NodeCountPerLayer` = {2, 2}:
  //  Tree
  //  |-Tree
  //  `-Tree
  //    |-Tree
  //    | `-'('
  //    `-Tree
  //      `-')'
  std::vector<const Tree *>
  generateAllTreesWithShape(ArrayRef<const Node *> Base,
                            ArrayRef<unsigned> NodeCountPerLayer) {
    // We compute the solution per layer. A layer is a collection of bases,
    // where each base has the same number of nodes, given by
    // `NodeCountPerLayer`.
    auto GenerateNextLayer = [this](ArrayRef<std::vector<const Node *>> Layer,
                                    unsigned NextLayerNodeCount) {
      std::vector<std::vector<const Node *>> NextLayer;
      for (const auto &Base : Layer) {
        for (const auto &NextBase :
             generateAllForests(Base, NextLayerNodeCount)) {
          NextLayer.push_back(
              std::vector<const Node *>(NextBase.begin(), NextBase.end()));
        }
      }
      return NextLayer;
    };

    std::vector<std::vector<const Node *>> Layer = {Base};
    for (auto NodeCount : NodeCountPerLayer)
      Layer = GenerateNextLayer(Layer, NodeCount);

    std::vector<const Tree *> AllTrees;
    AllTrees.reserve(Layer.size());
    for (const auto &Base : Layer)
      AllTrees.push_back(createTree(Base));

    return AllTrees;
  }
};

INSTANTIATE_TEST_CASE_P(TreeTests, TreeTest,
                        ::testing::ValuesIn(allTestClangConfigs()), );

TEST_P(TreeTest, FirstLeaf) {
  buildTree("", GetParam());
  std::vector<const Node *> Leafs = {createLeaf(*Arena, tok::l_paren),
                                     createLeaf(*Arena, tok::r_paren)};
  for (const auto *Tree : generateAllTreesWithShape(Leafs, {3u})) {
    ASSERT_TRUE(Tree->findFirstLeaf() != nullptr);
    EXPECT_EQ(Tree->findFirstLeaf()->getToken()->kind(), tok::l_paren);
  }
}

TEST_P(TreeTest, LastLeaf) {
  buildTree("", GetParam());
  std::vector<const Node *> Leafs = {createLeaf(*Arena, tok::l_paren),
                                     createLeaf(*Arena, tok::r_paren)};
  for (const auto *Tree : generateAllTreesWithShape(Leafs, {3u})) {
    ASSERT_TRUE(Tree->findLastLeaf() != nullptr);
    EXPECT_EQ(Tree->findLastLeaf()->getToken()->kind(), tok::r_paren);
  }
}

} // namespace
