//===-- Unittests for insque ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/search/insque.h"
#include "src/search/remque.h"
#include "test/UnitTest/Test.h"

namespace {

struct Node {
  Node *next;
  Node *prev;
};

template <unsigned N> void make_linear_links(Node (&nodes)[N]) {
  for (unsigned i = 0; i < N; ++i) {
    if (i == N - 1)
      nodes[i].next = nullptr;
    else
      nodes[i].next = &nodes[i + 1];

    if (i == 0)
      nodes[i].prev = nullptr;
    else
      nodes[i].prev = &nodes[i - 1];
  }
}

template <unsigned N> void make_circular_links(Node (&nodes)[N]) {
  for (unsigned i = 0; i < N; ++i) {
    nodes[i].next = &nodes[(i + 1) % N];
    nodes[i].prev = &nodes[(i + N - 1) % N];
  }
}

} // namespace

class LlvmLibcInsqueTest : public LIBC_NAMESPACE::testing::Test {
protected:
  template <unsigned N>
  void check_linear(const Node *head, const Node *const (&nodes)[N]) {
    // First make sure that the given N nodes form a valid linear list.
    for (unsigned i = 0; i < N; ++i) {
      const Node *next = nullptr;
      if (i + 1 < N)
        next = nodes[i + 1];

      const Node *prev = nullptr;
      if (i > 0)
        prev = nodes[i - 1];

      EXPECT_EQ(static_cast<const Node *>(nodes[i]->next), next);
      EXPECT_EQ(static_cast<const Node *>(nodes[i]->prev), prev);
    }

    // Then check the list nodes match.
    for (unsigned i = 0; i < N; ++i) {
      EXPECT_EQ(head, nodes[i]);
      // Traversal by head should always be OK since we have already confirmed
      // the validity of links.
      head = head->next;
    }
  }

  template <unsigned N>
  void check_circular(const Node *head, const Node *const (&nodes)[N]) {
    // First make sure that the given N nodes form a valid linear list.
    for (unsigned i = 0; i < N; ++i) {
      auto next = nodes[(i + 1) % N];
      auto prev = nodes[(i + N - 1) % N];

      EXPECT_EQ(static_cast<const Node *>(nodes[i]->prev), prev);
      EXPECT_EQ(static_cast<const Node *>(nodes[i]->next), next);
    }

    // Then check the list nodes match.
    for (unsigned i = 0; i < N; ++i) {
      EXPECT_EQ(head, nodes[i]);
      // Traversal by head should always be OK since we have already confirmed
      // the validity of links.
      head = head->next;
    }
  }
};

TEST_F(LlvmLibcInsqueTest, InsertToNull) {
  Node node{nullptr, nullptr};
  LIBC_NAMESPACE::insque(&node, nullptr);
  check_linear(&node, {&node});
}

TEST_F(LlvmLibcInsqueTest, InsertToLinearSingleton) {
  Node base[1];
  make_linear_links(base);

  Node incoming{nullptr, nullptr};
  LIBC_NAMESPACE::insque(&incoming, &base[0]);
  check_linear(base, {&base[0], &incoming});
}

TEST_F(LlvmLibcInsqueTest, InsertToLinearMiddle) {
  Node base[3];
  make_linear_links(base);

  Node incoming{nullptr, nullptr};
  LIBC_NAMESPACE::insque(&incoming, &base[1]);
  check_linear(base, {&base[0], &base[1], &incoming, &base[2]});
}

TEST_F(LlvmLibcInsqueTest, InsertToLinearBack) {
  Node base[3];
  make_linear_links(base);

  Node incoming{nullptr, nullptr};
  LIBC_NAMESPACE::insque(&incoming, &base[2]);
  check_linear(base, {&base[0], &base[1], &base[2], &incoming});
}

TEST_F(LlvmLibcInsqueTest, InsertToCircularSingleton) {
  Node base[1];
  make_circular_links(base);

  Node incoming{nullptr, nullptr};
  LIBC_NAMESPACE::insque(&incoming, &base[0]);
  check_circular(base, {&base[0], &incoming});
}

TEST_F(LlvmLibcInsqueTest, InsertToCircular) {
  Node base[3];
  make_circular_links(base);

  Node incoming{nullptr, nullptr};
  LIBC_NAMESPACE::insque(&incoming, &base[1]);
  check_circular(base, {&base[0], &base[1], &incoming, &base[2]});
}

TEST_F(LlvmLibcInsqueTest, RemoveFromLinearSingleton) {
  Node node{nullptr, nullptr};
  LIBC_NAMESPACE::remque(&node);
  ASSERT_EQ(node.next, static_cast<Node *>(nullptr));
  ASSERT_EQ(node.prev, static_cast<Node *>(nullptr));
}

TEST_F(LlvmLibcInsqueTest, RemoveFromLinearFront) {
  Node base[3];
  make_linear_links(base);

  LIBC_NAMESPACE::remque(&base[0]);
  check_linear(&base[1], {&base[1], &base[2]});
}

TEST_F(LlvmLibcInsqueTest, RemoveFromLinearMiddle) {
  Node base[3];
  make_linear_links(base);

  LIBC_NAMESPACE::remque(&base[1]);
  check_linear(&base[0], {&base[0], &base[2]});
}

TEST_F(LlvmLibcInsqueTest, RemoveFromLinearBack) {
  Node base[3];
  make_linear_links(base);

  LIBC_NAMESPACE::remque(&base[2]);
  check_linear(&base[0], {&base[0], &base[1]});
}

TEST_F(LlvmLibcInsqueTest, RemoveFromCircularSingleton) {
  Node node[1];
  make_circular_links(node);

  LIBC_NAMESPACE::remque(&node[0]);
}

TEST_F(LlvmLibcInsqueTest, RemoveFromCircular) {
  Node base[3];
  make_circular_links(base);

  LIBC_NAMESPACE::remque(&base[1]);
  check_circular(&base[0], {&base[0], &base[2]});
}
