//===- TreeTest.cpp ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Syntax/Tree.h"
#include "TreeTestBase.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Tooling/Syntax/BuildTree.h"
#include "clang/Tooling/Syntax/Nodes.h"
#include "llvm/ADT/STLExtras.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace clang::syntax;

namespace {
using testing::ElementsAre;

class TreeTest : public SyntaxTreeTest {
private:
  Tree *createTree(ArrayRef<const Node *> Children) {
    std::vector<std::pair<Node *, NodeRole>> ChildrenWithRoles;
    ChildrenWithRoles.reserve(Children.size());
    for (const auto *Child : Children) {
      ChildrenWithRoles.push_back(std::make_pair(
          deepCopyExpandingMacros(*Arena, Child), NodeRole::Unknown));
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

INSTANTIATE_TEST_SUITE_P(TreeTests, TreeTest,
                        ::testing::ValuesIn(allTestClangConfigs()) );

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

TEST_F(TreeTest, Iterators) {
  buildTree("", allTestClangConfigs().front());
  std::vector<Node *> Children = {createLeaf(*Arena, tok::identifier, "a"),
                                  createLeaf(*Arena, tok::identifier, "b"),
                                  createLeaf(*Arena, tok::identifier, "c")};
  auto *Tree = syntax::createTree(*Arena,
                                  {{Children[0], NodeRole::LeftHandSide},
                                   {Children[1], NodeRole::OperatorToken},
                                   {Children[2], NodeRole::RightHandSide}},
                                  NodeKind::TranslationUnit);
  const auto *ConstTree = Tree;

  auto Range = Tree->getChildren();
  EXPECT_THAT(Range, ElementsAre(role(NodeRole::LeftHandSide),
                                 role(NodeRole::OperatorToken),
                                 role(NodeRole::RightHandSide)));

  auto ConstRange = ConstTree->getChildren();
  EXPECT_THAT(ConstRange, ElementsAre(role(NodeRole::LeftHandSide),
                                      role(NodeRole::OperatorToken),
                                      role(NodeRole::RightHandSide)));

  // FIXME: mutate and observe no invalidation. Mutations are private for now...
  auto It = Range.begin();
  auto CIt = ConstRange.begin();
  static_assert(std::is_same<decltype(*It), syntax::Node &>::value,
                "mutable range");
  static_assert(std::is_same<decltype(*CIt), const syntax::Node &>::value,
                "const range");

  for (unsigned I = 0; I < 3; ++I) {
    EXPECT_EQ(It, CIt);
    EXPECT_TRUE(It);
    EXPECT_TRUE(CIt);
    EXPECT_EQ(It.asPointer(), Children[I]);
    EXPECT_EQ(CIt.asPointer(), Children[I]);
    EXPECT_EQ(&*It, Children[I]);
    EXPECT_EQ(&*CIt, Children[I]);
    ++It;
    ++CIt;
  }
  EXPECT_EQ(It, CIt);
  EXPECT_EQ(It, Tree::ChildIterator());
  EXPECT_EQ(CIt, Tree::ConstChildIterator());
  EXPECT_FALSE(It);
  EXPECT_FALSE(CIt);
  EXPECT_EQ(nullptr, It.asPointer());
  EXPECT_EQ(nullptr, CIt.asPointer());
}

class ListTest : public SyntaxTreeTest {
private:
  std::string dumpQuotedTokensOrNull(const Node *N) {
    return N ? "'" +
                   StringRef(N->dumpTokens(Arena->getSourceManager()))
                       .trim()
                       .str() +
                   "'"
             : "null";
  }

protected:
  std::string
  dumpElementsAndDelimiters(ArrayRef<List::ElementAndDelimiter<Node>> EDs) {
    std::string Storage;
    llvm::raw_string_ostream OS(Storage);

    OS << "[";

    llvm::interleaveComma(
        EDs, OS, [&OS, this](const List::ElementAndDelimiter<Node> &ED) {
          OS << "(" << dumpQuotedTokensOrNull(ED.element) << ", "
             << dumpQuotedTokensOrNull(ED.delimiter) << ")";
        });

    OS << "]";

    return Storage;
  }

  std::string dumpNodes(ArrayRef<Node *> Nodes) {
    std::string Storage;
    llvm::raw_string_ostream OS(Storage);

    OS << "[";

    llvm::interleaveComma(Nodes, OS, [&OS, this](const Node *N) {
      OS << dumpQuotedTokensOrNull(N);
    });

    OS << "]";

    return Storage;
  }
};

INSTANTIATE_TEST_SUITE_P(TreeTests, ListTest,
                        ::testing::ValuesIn(allTestClangConfigs()) );

/// "a, b, c"  <=> [("a", ","), ("b", ","), ("c", null)]
TEST_P(ListTest, List_Separated_WellFormed) {
  buildTree("", GetParam());

  // "a, b, c"
  auto *List = dyn_cast<syntax::List>(syntax::createTree(
      *Arena,
      {
          {createLeaf(*Arena, tok::identifier, "a"), NodeRole::ListElement},
          {createLeaf(*Arena, tok::comma), NodeRole::ListDelimiter},
          {createLeaf(*Arena, tok::identifier, "b"), NodeRole::ListElement},
          {createLeaf(*Arena, tok::comma), NodeRole::ListDelimiter},
          {createLeaf(*Arena, tok::identifier, "c"), NodeRole::ListElement},
      },
      NodeKind::CallArguments));

  EXPECT_EQ(dumpElementsAndDelimiters(List->getElementsAsNodesAndDelimiters()),
            "[('a', ','), ('b', ','), ('c', null)]");
  EXPECT_EQ(dumpNodes(List->getElementsAsNodes()), "['a', 'b', 'c']");
}

/// "a,  , c"  <=> [("a", ","), (null, ","), ("c", null)]
TEST_P(ListTest, List_Separated_MissingElement) {
  buildTree("", GetParam());

  // "a,  , c"
  auto *List = dyn_cast<syntax::List>(syntax::createTree(
      *Arena,
      {
          {createLeaf(*Arena, tok::identifier, "a"), NodeRole::ListElement},
          {createLeaf(*Arena, tok::comma), NodeRole::ListDelimiter},
          {createLeaf(*Arena, tok::comma), NodeRole::ListDelimiter},
          {createLeaf(*Arena, tok::identifier, "c"), NodeRole::ListElement},
      },
      NodeKind::CallArguments));

  EXPECT_EQ(dumpElementsAndDelimiters(List->getElementsAsNodesAndDelimiters()),
            "[('a', ','), (null, ','), ('c', null)]");
  EXPECT_EQ(dumpNodes(List->getElementsAsNodes()), "['a', null, 'c']");
}

/// "a, b  c"  <=> [("a", ","), ("b", null), ("c", null)]
TEST_P(ListTest, List_Separated_MissingDelimiter) {
  buildTree("", GetParam());

  // "a, b  c"
  auto *List = dyn_cast<syntax::List>(syntax::createTree(
      *Arena,
      {
          {createLeaf(*Arena, tok::identifier, "a"), NodeRole::ListElement},
          {createLeaf(*Arena, tok::comma), NodeRole::ListDelimiter},
          {createLeaf(*Arena, tok::identifier, "b"), NodeRole::ListElement},
          {createLeaf(*Arena, tok::identifier, "c"), NodeRole::ListElement},
      },
      NodeKind::CallArguments));

  EXPECT_EQ(dumpElementsAndDelimiters(List->getElementsAsNodesAndDelimiters()),
            "[('a', ','), ('b', null), ('c', null)]");
  EXPECT_EQ(dumpNodes(List->getElementsAsNodes()), "['a', 'b', 'c']");
}

/// "a, b,"    <=> [("a", ","), ("b", ","), (null, null)]
TEST_P(ListTest, List_Separated_MissingLastElement) {
  buildTree("", GetParam());

  // "a, b, c"
  auto *List = dyn_cast<syntax::List>(syntax::createTree(
      *Arena,
      {
          {createLeaf(*Arena, tok::identifier, "a"), NodeRole::ListElement},
          {createLeaf(*Arena, tok::comma), NodeRole::ListDelimiter},
          {createLeaf(*Arena, tok::identifier, "b"), NodeRole::ListElement},
          {createLeaf(*Arena, tok::comma), NodeRole::ListDelimiter},
      },
      NodeKind::CallArguments));

  EXPECT_EQ(dumpElementsAndDelimiters(List->getElementsAsNodesAndDelimiters()),
            "[('a', ','), ('b', ','), (null, null)]");
  EXPECT_EQ(dumpNodes(List->getElementsAsNodes()), "['a', 'b', null]");
}

/// "a:: b:: c::" <=> [("a", "::"), ("b", "::"), ("c", "::")]
TEST_P(ListTest, List_Terminated_WellFormed) {
  if (!GetParam().isCXX()) {
    return;
  }
  buildTree("", GetParam());

  // "a:: b:: c::"
  auto *List = dyn_cast<syntax::List>(syntax::createTree(
      *Arena,
      {
          {createLeaf(*Arena, tok::identifier, "a"), NodeRole::ListElement},
          {createLeaf(*Arena, tok::coloncolon), NodeRole::ListDelimiter},
          {createLeaf(*Arena, tok::identifier, "b"), NodeRole::ListElement},
          {createLeaf(*Arena, tok::coloncolon), NodeRole::ListDelimiter},
          {createLeaf(*Arena, tok::identifier, "c"), NodeRole::ListElement},
          {createLeaf(*Arena, tok::coloncolon), NodeRole::ListDelimiter},
      },
      NodeKind::NestedNameSpecifier));

  EXPECT_EQ(dumpElementsAndDelimiters(List->getElementsAsNodesAndDelimiters()),
            "[('a', '::'), ('b', '::'), ('c', '::')]");
  EXPECT_EQ(dumpNodes(List->getElementsAsNodes()), "['a', 'b', 'c']");
}

/// "a::  :: c::" <=> [("a", "::"), (null, "::"), ("c", "::")]
TEST_P(ListTest, List_Terminated_MissingElement) {
  if (!GetParam().isCXX()) {
    return;
  }
  buildTree("", GetParam());

  // "a:: b:: c::"
  auto *List = dyn_cast<syntax::List>(syntax::createTree(
      *Arena,
      {
          {createLeaf(*Arena, tok::identifier, "a"), NodeRole::ListElement},
          {createLeaf(*Arena, tok::coloncolon), NodeRole::ListDelimiter},
          {createLeaf(*Arena, tok::coloncolon), NodeRole::ListDelimiter},
          {createLeaf(*Arena, tok::identifier, "c"), NodeRole::ListElement},
          {createLeaf(*Arena, tok::coloncolon), NodeRole::ListDelimiter},
      },
      NodeKind::NestedNameSpecifier));

  EXPECT_EQ(dumpElementsAndDelimiters(List->getElementsAsNodesAndDelimiters()),
            "[('a', '::'), (null, '::'), ('c', '::')]");
  EXPECT_EQ(dumpNodes(List->getElementsAsNodes()), "['a', null, 'c']");
}

/// "a:: b  c::" <=> [("a", "::"), ("b", null), ("c", "::")]
TEST_P(ListTest, List_Terminated_MissingDelimiter) {
  if (!GetParam().isCXX()) {
    return;
  }
  buildTree("", GetParam());

  // "a:: b  c::"
  auto *List = dyn_cast<syntax::List>(syntax::createTree(
      *Arena,
      {
          {createLeaf(*Arena, tok::identifier, "a"), NodeRole::ListElement},
          {createLeaf(*Arena, tok::coloncolon), NodeRole::ListDelimiter},
          {createLeaf(*Arena, tok::identifier, "b"), NodeRole::ListElement},
          {createLeaf(*Arena, tok::identifier, "c"), NodeRole::ListElement},
          {createLeaf(*Arena, tok::coloncolon), NodeRole::ListDelimiter},
      },
      NodeKind::NestedNameSpecifier));

  EXPECT_EQ(dumpElementsAndDelimiters(List->getElementsAsNodesAndDelimiters()),
            "[('a', '::'), ('b', null), ('c', '::')]");
  EXPECT_EQ(dumpNodes(List->getElementsAsNodes()), "['a', 'b', 'c']");
}

/// "a:: b:: c"  <=> [("a", "::"), ("b", "::"), ("c", null)]
TEST_P(ListTest, List_Terminated_MissingLastDelimiter) {
  if (!GetParam().isCXX()) {
    return;
  }
  buildTree("", GetParam());

  // "a:: b:: c"
  auto *List = dyn_cast<syntax::List>(syntax::createTree(
      *Arena,
      {
          {createLeaf(*Arena, tok::identifier, "a"), NodeRole::ListElement},
          {createLeaf(*Arena, tok::coloncolon), NodeRole::ListDelimiter},
          {createLeaf(*Arena, tok::identifier, "b"), NodeRole::ListElement},
          {createLeaf(*Arena, tok::coloncolon), NodeRole::ListDelimiter},
          {createLeaf(*Arena, tok::identifier, "c"), NodeRole::ListElement},
      },
      NodeKind::NestedNameSpecifier));

  EXPECT_EQ(dumpElementsAndDelimiters(List->getElementsAsNodesAndDelimiters()),
            "[('a', '::'), ('b', '::'), ('c', null)]");
  EXPECT_EQ(dumpNodes(List->getElementsAsNodes()), "['a', 'b', 'c']");
}

} // namespace
