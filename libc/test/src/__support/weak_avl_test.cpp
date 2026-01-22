//===-- Unittests for WeakAVL ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/optional.h"
#include "src/__support/weak_avl.h"
#include "test/UnitTest/Test.h"

using Node = LIBC_NAMESPACE::WeakAVLNode<int>;

namespace {
int ternary_compare(int a, int b) { return (a > b) - (a < b); }
constexpr int TEST_SIZE = 128;
// Validate weak-AVL rank-difference invariant assuming **pure insertion only**
// (i.e. no erasure has occurred).
//
// NOTE: This validator is intentionally *not* correct after erase(), because
// weak-AVL allows transient or permanent 2-2 configurations during deletion
// fixup.
bool validate_pure_insertion(const Node *node) {
  if (!node)
    return true;
  bool left_2 = node->has_rank_diff_2(false);
  bool right_2 = node->has_rank_diff_2(true);
  return (!left_2 || !right_2) && validate_pure_insertion(node->get_left()) &&
         validate_pure_insertion(node->get_right());
}

// Insert according to pattern `next(i)`
using NextFn = int (*)(int);
using OptionalNodePtr = LIBC_NAMESPACE::cpp::optional<Node *>;
struct Tree {
  Node *root = nullptr;

  bool validate_pure_insertion() { return ::validate_pure_insertion(root); }

  bool contains(int value) {
    return Node::find(root, value, ternary_compare).has_value();
  }

  OptionalNodePtr insert(int value) {
    return Node::find_or_insert(root, value, ternary_compare);
  }

  OptionalNodePtr find(int value) {
    return Node::find(root, value, ternary_compare);
  }

  void erase(int value) {
    if (OptionalNodePtr node = Node::find(root, value, ternary_compare))
      Node::erase(root, node.value());
  }

  template <typename NextFn> static Tree build(NextFn next, int N) {
    Tree tree;
    for (int i = 0; i < N; ++i)
      tree.insert(next(i));
    return tree;
  }

  bool empty() const { return root == nullptr; }

  ~Tree() { Node::destroy(root); }
};

// Insertion patterns
static int seq(int i) { return i; }

static int rev(int i) {
  constexpr int N = TEST_SIZE;
  return N - 1 - i;
}

// Coprime stride permutation: i -> (i * X) % N
static int stride(int i, int prime = 7919) {
  constexpr int N = TEST_SIZE;
  return (i * prime) % N;
}

} // namespace

TEST(LlvmLibcWeakAVLTest, SimpleInsertion) {
  Tree tree;

  OptionalNodePtr node10 = tree.insert(10);
  ASSERT_TRUE(node10.has_value());
  ASSERT_TRUE(tree.insert(5).has_value());
  ASSERT_TRUE(tree.validate_pure_insertion());

  OptionalNodePtr node15 = tree.insert(15);
  ASSERT_TRUE(node15.has_value());
  ASSERT_TRUE(tree.validate_pure_insertion());

  OptionalNodePtr node10_again = tree.insert(10);
  ASSERT_EQ(*node10, *node10_again);
  ASSERT_TRUE(tree.validate_pure_insertion());
}

TEST(LlvmLibcWeakAVLTest, SequentialInsertion) {
  constexpr int N = TEST_SIZE;

  Tree tree = Tree::build(seq, N);
  ASSERT_TRUE(tree.validate_pure_insertion());

  for (int i = 0; i < N; ++i) {
    OptionalNodePtr node = tree.insert(i);
    ASSERT_TRUE(node.has_value());
    ASSERT_EQ(node.value()->get_data(), i);
  }

  ASSERT_TRUE(tree.validate_pure_insertion());
}

TEST(LlvmLibcWeakAVLTest, ReversedInsertion) {
  constexpr int N = TEST_SIZE;

  Tree tree = Tree::build(rev, N);
  ASSERT_TRUE(tree.validate_pure_insertion());

  for (int i = 0; i < N; ++i) {
    OptionalNodePtr node = tree.insert(i);
    ASSERT_TRUE(node.has_value());
    ASSERT_EQ(node.value()->get_data(), i);
  }

  ASSERT_TRUE(tree.validate_pure_insertion());
}

TEST(LlvmLibcWeakAVLTest, StridedInsertion) {
  constexpr int N = TEST_SIZE;

  Tree tree = Tree::build([](int i) { return stride(i); }, N);
  ASSERT_TRUE(tree.validate_pure_insertion());

  for (int i = 0; i < N; ++i) {
    OptionalNodePtr node = tree.insert(i);
    ASSERT_TRUE(node.has_value());
    ASSERT_EQ(node.value()->get_data(), i);
  }

  ASSERT_TRUE(tree.validate_pure_insertion());
}

TEST(LlvmLibcWeakAVLTest, FindExistingAndMissing) {
  constexpr int N = TEST_SIZE;

  Tree tree = Tree::build(seq, N);
  ASSERT_TRUE(tree.validate_pure_insertion());

  for (int i = 0; i < N; ++i) {
    OptionalNodePtr node = tree.find(i);
    ASSERT_TRUE(node.has_value());
    ASSERT_EQ(node.value()->get_data(), i);
  }

  ASSERT_FALSE(tree.find(-1).has_value());
  ASSERT_FALSE(tree.find(N).has_value());
  ASSERT_FALSE(tree.find(2 * N).has_value());
}

TEST(LlvmLibcWeakAVLTest, SequentialErase) {
  constexpr int N = TEST_SIZE;

  Tree tree = Tree::build(seq, N);

  for (int i = 0; i < N; ++i) {
    ASSERT_TRUE(tree.contains(i));
    tree.erase(i);
    ASSERT_FALSE(tree.contains(i));
  }

  ASSERT_TRUE(tree.empty());
}

TEST(LlvmLibcWeakAVLTest, ReverseErase) {
  constexpr int N = TEST_SIZE;

  Tree tree = Tree::build(seq, N);

  for (int i = N - 1; i >= 0; --i) {
    ASSERT_TRUE(tree.contains(i));
    tree.erase(i);
    ASSERT_FALSE(tree.contains(i));
  }

  ASSERT_TRUE(tree.empty());
}

TEST(LlvmLibcWeakAVLTest, StridedErase) {
  constexpr int N = TEST_SIZE;

  Tree tree = Tree::build(seq, N);

  for (int i = 0; i < N; ++i) {
    int key = stride(i, 5261);
    ASSERT_TRUE(tree.contains(key));
    tree.erase(key);
    ASSERT_FALSE(tree.contains(key));
  }

  ASSERT_TRUE(tree.empty());
}

TEST(LlvmLibcWeakAVLTest, EraseStructuralCases) {
  Tree tree;
  int keys[] = {10, 5, 15, 3, 7, 12, 18};

  // rank1:               10              10
  //                      /              / \
  // rank0:   10   -->   5        -->   5   15

  // rank2:                   10               10
  //                         / \              / \
  // rank1:   10            5   \            5   \
  //         / \   -->     /     \     -->  /\    \
  // rank0: 5  15         3       15       3  7    15

  // rank2:     10            10             10
  //           / \           / \            / \
  // rank1:   5   \   -->   5   15   -->   5   15
  //         /\    \       /\   /         /\   / \
  // rank0: 3  7    15    3  7 12       3  7 12  18

  for (int k : keys)
    tree.insert(k);

  // Erase leaf.
  // rank2:     10                   10
  //           / \                  / \
  // rank1:   5   15               5   15
  //         /\   / \      -->     \   / \
  // rank0: 3  7 12  18            7  12  18
  tree.erase(3);
  ASSERT_FALSE(tree.contains(3));

  // Erase internal nodes.
  // Erase leaf.
  // rank2:     10                   10               10
  //           / \                  / \              / \
  // rank1:   5   15               7   15           /  15
  //          \   / \      -->     \   / \    -->  /   /\
  // rank0:    7 12  18            5  12  18      7  12  18
  tree.erase(5);
  ASSERT_FALSE(tree.contains(5));

  // Erase root.
  // rank2:     10              12               12
  //           / \             / \              / \
  // rank1:   /  15     -->   /  15     -->    /  15
  //         /   /\          /   /\           /    \
  // rank0: 7  12  18      7  10  18         7     18
  tree.erase(10);
  ASSERT_FALSE(tree.contains(10));

  int attempts[] = {7, 12, 15, 18};
  for (int k : attempts)
    ASSERT_TRUE(tree.contains(k));
}

TEST(LlvmLibcTreeWalk, InOrderTraversal) {
  Tree tree = Tree::build([](int x) { return stride(x, 1007); }, TEST_SIZE);
  int data[TEST_SIZE];
  int counter = 0;
  Node::walk(tree.root, [&](Node *node, Node::WalkType type) {
    if (type == Node::WalkType::InOrder || type == Node::WalkType::Leaf)
      data[counter++] = node->get_data();
  });
  for (int i = 0; i < TEST_SIZE; ++i)
    ASSERT_EQ(data[i], i);
}
