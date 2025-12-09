//===- unittests/Analysis/Scalable/EntityIdTest.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/Model/EntityId.h"
#include "clang/Analysis/Scalable/Model/EntityIdTable.h"
#include "clang/Analysis/Scalable/Model/EntityName.h"
#include "gtest/gtest.h"

namespace clang {
namespace ssaf {
namespace {

TEST(EntityIdTest, Equality) {
  EntityIdTable Table;

  EntityName Entity1("c:@F@foo", "", {});
  EntityName Entity2("c:@F@bar", "", {});

  EntityId Id1 = Table.createEntityId(Entity1);
  EntityId Id2 = Table.createEntityId(Entity2);
  EntityId Id1Copy = Table.createEntityId(Entity1);

  EXPECT_EQ(Id1, Id1Copy);
  EXPECT_FALSE(Id1 != Id1Copy);

  EXPECT_NE(Id1, Id2);
  EXPECT_FALSE(Id1 == Id2);
}

TEST(EntityIdTest, LessThan) {
  EntityIdTable Table;

  EntityName Entity1("c:@F@aaa", "", {});
  EntityName Entity2("c:@F@bbb", "", {});
  EntityName Entity3("c:@F@ccc", "", {});

  EntityId Id1 = Table.createEntityId(Entity1);
  EntityId Id2 = Table.createEntityId(Entity2);
  EntityId Id3 = Table.createEntityId(Entity3);

  EXPECT_TRUE(Id1 < Id2 || Id2 < Id1);
  EXPECT_TRUE(Id1 < Id3 || Id3 < Id1);
  EXPECT_TRUE(Id2 < Id3 || Id3 < Id2);

  // Transitivity: if a < b and b < c, then a < c
  if (Id1 < Id2 && Id2 < Id3) {
    EXPECT_TRUE(Id1 < Id3);
  }
}

} // namespace
} // namespace ssaf
} // namespace clang
