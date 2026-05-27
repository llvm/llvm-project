//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for flat_tlsf Node.
///
//===----------------------------------------------------------------------===//

#include "src/__support/flat_tlsf/node.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::flat_tlsf::Node;

TEST(LlvmLibcFlatTlsfNodeTest, BasicOperations) {
  Node x{nullptr, nullptr};
  Node y{nullptr, nullptr};
  Node z{nullptr, nullptr};

  y.link_at(Node{nullptr, x.addr_of_next()});
  z.link_at(Node{&y, x.addr_of_next()});

  {
    Node *curr = &x;
    ASSERT_NE(curr, static_cast<Node *>(nullptr));
    EXPECT_EQ(curr, &x);
    curr = curr->next;
    ASSERT_NE(curr, static_cast<Node *>(nullptr));
    EXPECT_EQ(curr, &z);
    curr = curr->next;
    ASSERT_NE(curr, static_cast<Node *>(nullptr));
    EXPECT_EQ(curr, &y);
    curr = curr->next;
    EXPECT_EQ(curr, static_cast<Node *>(nullptr));
  }

  {
    Node *curr = &y;
    ASSERT_NE(curr, static_cast<Node *>(nullptr));
    EXPECT_EQ(curr, &y);
    curr = curr->next;
    EXPECT_EQ(curr, static_cast<Node *>(nullptr));
  }

  z.unlink();

  {
    Node *curr = &x;
    ASSERT_NE(curr, static_cast<Node *>(nullptr));
    EXPECT_EQ(curr, &x);
    curr = curr->next;
    ASSERT_NE(curr, static_cast<Node *>(nullptr));
    EXPECT_EQ(curr, &y);
    curr = curr->next;
    EXPECT_EQ(curr, static_cast<Node *>(nullptr));
  }

  z.link_at(Node{&y, x.addr_of_next()});

  {
    Node *curr = &x;
    ASSERT_NE(curr, static_cast<Node *>(nullptr));
    EXPECT_EQ(curr, &x);
    curr = curr->next;
    ASSERT_NE(curr, static_cast<Node *>(nullptr));
    EXPECT_EQ(curr, &z);
    curr = curr->next;
    ASSERT_NE(curr, static_cast<Node *>(nullptr));
    EXPECT_EQ(curr, &y);
    curr = curr->next;
    EXPECT_EQ(curr, static_cast<Node *>(nullptr));
  }

  z.unlink();
  y.unlink();

  {
    Node *curr = &x;
    ASSERT_NE(curr, static_cast<Node *>(nullptr));
    EXPECT_EQ(curr, &x);
    curr = curr->next;
    EXPECT_EQ(curr, static_cast<Node *>(nullptr));
  }
}
