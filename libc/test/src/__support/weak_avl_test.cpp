//===-- Unittests for WeakAVL ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/weak_avl.h"
#include "test/UnitTest/Test.h"

using Node = LIBC_NAMESPACE::WeakAVLNode<int>;

namespace {
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

static Node *build_tree(NextFn next, int N, int (*compare)(int, int)) {
  Node *root = nullptr;
  for (int i = 0; i < N; ++i)
    Node::find_or_insert(root, next(i), compare);
  return root;
}

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

// Thin wrappers to make test intent explicit.
template <typename Compare>
static Node *find(Node *root, int value, Compare &&comp) {
  return Node::find(root, value, comp);
}

static void erase(Node *&root, Node *node) { Node::erase(root, node); }

} // namespace

TEST(LlvmLibcWeakAVLTest, SimpleInsertion) {
  Node *root = nullptr;
  auto compare = [](int a, int b) { return (a > b) - (a < b); };

  Node *node10 = Node::find_or_insert(root, 10, compare);
  ASSERT_TRUE(node10 != nullptr);
  ASSERT_EQ(root, node10);
  ASSERT_TRUE(validate_pure_insertion(root));

  Node *node5 = Node::find_or_insert(root, 5, compare);
  ASSERT_TRUE(node5 != nullptr);
  ASSERT_TRUE(validate_pure_insertion(root));

  Node *node15 = Node::find_or_insert(root, 15, compare);
  ASSERT_TRUE(node15 != nullptr);
  ASSERT_TRUE(validate_pure_insertion(root));

  Node *node10_again = Node::find_or_insert(root, 10, compare);
  ASSERT_EQ(node10, node10_again);
  ASSERT_TRUE(validate_pure_insertion(root));

  Node::destroy(root);
}

TEST(LlvmLibcWeakAVLTest, SequentialInsertion) {
  auto compare = [](int a, int b) { return (a > b) - (a < b); };
  constexpr int N = TEST_SIZE;

  Node *root = build_tree(seq, N, compare);
  ASSERT_TRUE(validate_pure_insertion(root));

  for (int i = 0; i < N; ++i) {
    Node *node = Node::find_or_insert(root, i, compare);
    ASSERT_TRUE(node != nullptr);
    ASSERT_EQ(node->get_data(), i);
  }

  ASSERT_TRUE(validate_pure_insertion(root));
  Node::destroy(root);
}

TEST(LlvmLibcWeakAVLTest, ReversedInsertion) {
  auto compare = [](int a, int b) { return (a > b) - (a < b); };
  constexpr int N = TEST_SIZE;

  Node *root = build_tree(rev, N, compare);
  ASSERT_TRUE(validate_pure_insertion(root));

  for (int i = 0; i < N; ++i) {
    Node *node = Node::find_or_insert(root, i, compare);
    ASSERT_TRUE(node != nullptr);
    ASSERT_EQ(node->get_data(), i);
  }

  ASSERT_TRUE(validate_pure_insertion(root));
  Node::destroy(root);
}

TEST(LlvmLibcWeakAVLTest, StridedInsertion) {
  auto compare = [](int a, int b) { return (a > b) - (a < b); };
  constexpr int N = TEST_SIZE;

  Node *root = build_tree([](int i) { return stride(i); }, N, compare);
  ASSERT_TRUE(validate_pure_insertion(root));

  for (int i = 0; i < N; ++i) {
    Node *node = Node::find_or_insert(root, i, compare);
    ASSERT_TRUE(node != nullptr);
    ASSERT_EQ(node->get_data(), i);
  }

  ASSERT_TRUE(validate_pure_insertion(root));
  Node::destroy(root);
}

TEST(LlvmLibcWeakAVLTest, FindExistingAndMissing) {
  auto compare = [](int a, int b) { return (a > b) - (a < b); };
  constexpr int N = TEST_SIZE;

  Node *root = build_tree(seq, N, compare);
  ASSERT_TRUE(validate_pure_insertion(root));

  for (int i = 0; i < N; ++i) {
    Node *node = find(root, i, compare);
    ASSERT_TRUE(node != nullptr);
    ASSERT_EQ(node->get_data(), i);
  }

  ASSERT_TRUE(find(root, -1, compare) == nullptr);
  ASSERT_TRUE(find(root, N, compare) == nullptr);
  ASSERT_TRUE(find(root, 2 * N, compare) == nullptr);

  Node::destroy(root);
}

TEST(LlvmLibcWeakAVLTest, SequentialErase) {
  auto compare = [](int a, int b) { return (a > b) - (a < b); };
  constexpr int N = TEST_SIZE;

  Node *root = build_tree(seq, N, compare);

  for (int i = 0; i < N; ++i) {
    Node *node = find(root, i, compare);
    ASSERT_TRUE(node != nullptr);

    erase(root, node);
    ASSERT_TRUE(find(root, i, compare) == nullptr);
  }

  ASSERT_TRUE(root == nullptr);
}

TEST(LlvmLibcWeakAVLTest, ReverseErase) {
  auto compare = [](int a, int b) { return (a > b) - (a < b); };
  constexpr int N = TEST_SIZE;

  Node *root = build_tree(seq, N, compare);

  for (int i = N - 1; i >= 0; --i) {
    Node *node = find(root, i, compare);
    ASSERT_TRUE(node != nullptr);

    erase(root, node);
    ASSERT_TRUE(find(root, i, compare) == nullptr);
  }

  ASSERT_TRUE(root == nullptr);
}

TEST(LlvmLibcWeakAVLTest, StridedErase) {
  auto compare = [](int a, int b) { return (a > b) - (a < b); };
  constexpr int N = TEST_SIZE;

  Node *root = build_tree(seq, N, compare);

  for (int i = 0; i < N; ++i) {
    int key = stride(i, 5261);
    Node *node = find(root, key, compare);
    ASSERT_TRUE(node != nullptr);

    erase(root, node);
    ASSERT_TRUE(find(root, key, compare) == nullptr);
  }

  ASSERT_TRUE(root == nullptr);
}

TEST(LlvmLibcWeakAVLTest, EraseStructuralCases) {
  auto compare = [](int a, int b) { return (a > b) - (a < b); };

  Node *root = nullptr;
  int keys[] = {10, 5, 15, 3, 7, 12, 18};

  for (int k : keys)
    Node::find_or_insert(root, k, compare);

  // Erase leaf.
  erase(root, find(root, 3, compare));
  ASSERT_TRUE(find(root, 3, compare) == nullptr);

  // Erase internal nodes.
  erase(root, find(root, 5, compare));
  ASSERT_TRUE(find(root, 5, compare) == nullptr);

  erase(root, find(root, 10, compare));
  ASSERT_TRUE(find(root, 10, compare) == nullptr);

  int attempts[] = {7, 12, 15, 18};
  for (int k : attempts) {
    Node *n = find(root, k, compare);
    ASSERT_TRUE(n != nullptr);
    ASSERT_EQ(n->get_data(), k);
  }

  Node::destroy(root);
}

TEST(LlvmLibcTreeWalk, EraseStructuralCases) {
  using WeakAVLNode = LIBC_NAMESPACE::WeakAVLNode<int>;
  auto compare = [](int a, int b) { return (a > b) - (a < b); };

  WeakAVLNode *root = nullptr;
  int keys[] = {10, 5, 15, 3, 7, 12, 18};

  for (int k : keys)
    WeakAVLNode::find_or_insert(root, k, compare);

  // Erase leaf.
  erase(root, find(root, 3, compare));
  ASSERT_TRUE(find(root, 3, compare) == nullptr);

  // Erase internal nodes.
  erase(root, find(root, 5, compare));
  ASSERT_TRUE(find(root, 5, compare) == nullptr);

  erase(root, find(root, 10, compare));
  ASSERT_TRUE(find(root, 10, compare) == nullptr);

  int attempts[] = {7, 12, 15, 18};
  for (int k : attempts) {
    WeakAVLNode *n = find(root, k, compare);
    ASSERT_TRUE(n != nullptr);
    ASSERT_EQ(n->get_data(), k);
  }

  WeakAVLNode::destroy(root);
}

TEST(LlvmLibcTreeWalk, InOrderTraversal) {
  using WeakAVLNode = LIBC_NAMESPACE::WeakAVLNode<int>;
  auto compare = [](int a, int b) { return (a > b) - (a < b); };

  WeakAVLNode *root =
      build_tree([](int x) { return stride(x, 1007); }, TEST_SIZE, compare);
  int data[TEST_SIZE];
  int counter = 0;
  WeakAVLNode::walk(root, [&](WeakAVLNode *node, WeakAVLNode::WalkType type) {
    if (type == WeakAVLNode::WalkType::InOrder ||
        type == WeakAVLNode::WalkType::Leaf)
      data[counter++] = node->get_data();
  });

  for (int i = 0; i < TEST_SIZE; ++i)
    ASSERT_EQ(data[i], i);
  WeakAVLNode::destroy(root);
}
