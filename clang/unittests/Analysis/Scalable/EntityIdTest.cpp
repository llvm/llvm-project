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

namespace clang::ssaf {
namespace {

TEST(EntityIdTest, Equality) {
  EntityIdTable Table;

  EntityName Entity1("c:@F@foo", "", {});
  EntityName Entity2("c:@F@bar", "", {});

  EntityId Id1 = Table.getId(Entity1);
  EntityId Id2 = Table.getId(Entity2);
  EntityId Id1Copy = Table.getId(Entity1);

  EXPECT_EQ(Id1, Id1Copy);
  EXPECT_FALSE(Id1 != Id1Copy);
  EXPECT_FALSE(Id1 < Id1Copy);
  EXPECT_FALSE(Id1Copy < Id1);

  EXPECT_NE(Id1, Id2);
  EXPECT_FALSE(Id1 == Id2);
  EXPECT_TRUE(Id1 < Id2 || Id2 < Id1);
}

TEST(EntityIdTest, LessThan) {
  EntityIdTable Table;

  EntityName Entity1("c:@F@aaa", "", {});
  EntityName Entity2("c:@F@bbb", "", {});

  EntityId Id1 = Table.getId(Entity1);
  EntityId Id2 = Table.getId(Entity2);

  EXPECT_TRUE(Id1 < Id2 || Id2 < Id1);
}

TEST(EntityIdTest, Transitivity) {
  EntityIdTable Table;

  EntityName Entity1("c:@F@xxx", "", {});
  EntityName Entity2("c:@F@yyy", "", {});
  EntityName Entity3("c:@F@zzz", "", {});

  EntityId Ids[3] = {Table.getId(Entity1),
                     Table.getId(Entity2),
                     Table.getId(Entity3)};

  std::sort(Ids, Ids + 3);

  EXPECT_TRUE(Ids[0] < Ids[1] && Ids[1] < Ids[2]);
}

} // namespace
} // namespace clang::ssaf
