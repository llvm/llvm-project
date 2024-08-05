//===-- Unittests for a freelist --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stddef.h>

#include "src/__support/freelist.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {

TEST(LlvmLibcFreeList, FreeList) {
  cpp::byte mem1[1024];
  optional<Block<> *> maybeBlock = Block<>::init(mem1);
  ASSERT_TRUE(maybeBlock.has_value());
  Block<> *block1 = *maybeBlock;

  cpp::byte mem2[1024];
  maybeBlock = Block<>::init(mem2);
  ASSERT_TRUE(maybeBlock.has_value());
  Block<> *block2 = *maybeBlock;

  FreeList list;
  list.push(block1);
  ASSERT_FALSE(list.empty());
  EXPECT_EQ(list.front(), block1);

  list.push(block2);
  EXPECT_EQ(list.front(), block1);

  list.pop();
  ASSERT_FALSE(list.empty());
  EXPECT_EQ(list.front(), block2);

  list.pop();
  ASSERT_TRUE(list.empty());

  list.push(block1);
  list.push(block2);
  list.remove(reinterpret_cast<FreeList::Node *>(block2->usable_space()));
  EXPECT_EQ(list.front(), block1);
  list.pop();
  ASSERT_TRUE(list.empty());
}

} // namespace LIBC_NAMESPACE_DECL
