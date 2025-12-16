//===- unittests/Analysis/Scalable/EntityIdTableTest.cpp -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/Model/EntityIdTable.h"
#include "clang/Analysis/Scalable/Model/BuildNamespace.h"
#include "clang/Analysis/Scalable/Model/EntityId.h"
#include "clang/Analysis/Scalable/Model/EntityName.h"
#include "gtest/gtest.h"

namespace clang {
namespace ssaf {
namespace {

TEST(EntityIdTableTest, CreateNewEntity) {
  EntityIdTable Table;

  EntityName Entity("c:@F@foo", "", {});
  Table.getId(Entity);

  EXPECT_TRUE(Table.contains(Entity));
}

TEST(EntityIdTableTest, Idempotency) {
  EntityIdTable Table;

  EntityName Entity("c:@F@foo", "", {});

  EntityId Id1 = Table.getId(Entity);
  EntityId Id2 = Table.getId(Entity);
  EntityId Id3 = Table.getId(Entity);

  EXPECT_EQ(Id1, Id2);
  EXPECT_EQ(Id2, Id3);
  EXPECT_EQ(Id1, Id3);
}

TEST(EntityIdTableTest, ExistsTrue) {
  EntityIdTable Table;

  EntityName Entity1("c:@F@foo", "", {});
  EntityName Entity2("c:@V@bar", "", {});

  Table.getId(Entity1);
  Table.getId(Entity2);

  EXPECT_TRUE(Table.contains(Entity1));
  EXPECT_TRUE(Table.contains(Entity2));
}

TEST(EntityIdTableTest, ExistsFalse) {
  EntityIdTable Table;

  EntityName Entity1("c:@F@foo", "", {});
  EntityName Entity2("c:@F@bar", "", {});

  Table.getId(Entity1);

  EXPECT_TRUE(Table.contains(Entity1));
  EXPECT_FALSE(Table.contains(Entity2));
}

TEST(EntityIdTableTest, MultipleEntities) {
  EntityIdTable Table;

  EntityName Entity1("c:@F@foo", "", {});
  EntityName Entity2("c:@F@bar", "", {});
  EntityName Entity3("c:@V@baz", "", {});

  EntityId Id1 = Table.getId(Entity1);
  EntityId Id2 = Table.getId(Entity2);
  EntityId Id3 = Table.getId(Entity3);

  EXPECT_NE(Id1, Id2);
  EXPECT_NE(Id1, Id3);
  EXPECT_NE(Id2, Id3);
}

TEST(EntityIdTableTest, WithBuildNamespace) {
  EntityIdTable Table;

  NestedBuildNamespace NS = NestedBuildNamespace::makeCompilationUnit("test.o");

  EntityName Entity1("c:@F@foo", "", NS);
  EntityName Entity2("c:@F@foo", "",
                     NestedBuildNamespace::makeCompilationUnit("other.o"));

  EntityId Id1 = Table.getId(Entity1);
  EntityId Id2 = Table.getId(Entity2);

  EXPECT_NE(Id1, Id2);
}

TEST(EntityIdTableTest, ForEachEmptyTable) {
  EntityIdTable Table;

  int CallbackCount = 0;
  Table.forEach(
      [&CallbackCount](const EntityName &, EntityId) { CallbackCount++; });

  EXPECT_EQ(CallbackCount, 0);
}

TEST(EntityIdTableTest, ForEachMultipleEntities) {
  EntityIdTable Table;

  EntityName Entity1("c:@F@foo", "", {});
  EntityName Entity2("c:@F@bar", "", {});
  EntityName Entity3("c:@V@baz", "", {});

  EntityId Id1 = Table.getId(Entity1);
  EntityId Id2 = Table.getId(Entity2);
  EntityId Id3 = Table.getId(Entity3);

  std::set<EntityId> VisitedIds;
  std::set<EntityName> VisitedNames;

  Table.forEach([&](const EntityName &Name, EntityId Id) {
    VisitedIds.insert(Id);
    VisitedNames.insert(Name);
  });

  EXPECT_EQ(VisitedIds.size(), 3u);
  EXPECT_EQ(VisitedNames.size(), 3u);

  EXPECT_TRUE(VisitedIds.count(Id1));
  EXPECT_TRUE(VisitedIds.count(Id2));
  EXPECT_TRUE(VisitedIds.count(Id3));

  EXPECT_TRUE(VisitedNames.count(Entity1));
  EXPECT_TRUE(VisitedNames.count(Entity2));
  EXPECT_TRUE(VisitedNames.count(Entity3));
}

} // namespace
} // namespace ssaf
} // namespace clang
