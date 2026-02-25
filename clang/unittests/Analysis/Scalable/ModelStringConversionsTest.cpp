//===- ModelStringConversionsTest.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ModelStringConversions.h"
#include "gtest/gtest.h"

using namespace clang::ssaf;

namespace {

//===----------------------------------------------------------------------===//
// BuildNamespaceKind
//===----------------------------------------------------------------------===//

TEST(BuildNamespaceKindStringTest, ToStringCompilationUnit) {
  EXPECT_EQ(buildNamespaceKindToString(BuildNamespaceKind::CompilationUnit),
            "CompilationUnit");
}

TEST(BuildNamespaceKindStringTest, ToStringLinkUnit) {
  EXPECT_EQ(buildNamespaceKindToString(BuildNamespaceKind::LinkUnit),
            "LinkUnit");
}

TEST(BuildNamespaceKindStringTest, FromStringCompilationUnit) {
  EXPECT_EQ(buildNamespaceKindFromString("CompilationUnit"),
            BuildNamespaceKind::CompilationUnit);
}

TEST(BuildNamespaceKindStringTest, FromStringLinkUnit) {
  EXPECT_EQ(buildNamespaceKindFromString("LinkUnit"),
            BuildNamespaceKind::LinkUnit);
}

TEST(BuildNamespaceKindStringTest, FromStringUnknown) {
  EXPECT_EQ(buildNamespaceKindFromString("compilation_unit"), std::nullopt);
  EXPECT_EQ(buildNamespaceKindFromString("link_unit"), std::nullopt);
  EXPECT_EQ(buildNamespaceKindFromString(""), std::nullopt);
  EXPECT_EQ(buildNamespaceKindFromString("unknown"), std::nullopt);
}

TEST(BuildNamespaceKindStringTest, RoundTrip) {
  for (auto Kind :
       {BuildNamespaceKind::CompilationUnit, BuildNamespaceKind::LinkUnit}) {
    EXPECT_EQ(buildNamespaceKindFromString(buildNamespaceKindToString(Kind)),
              Kind);
  }
}

//===----------------------------------------------------------------------===//
// EntityLinkageType
//===----------------------------------------------------------------------===//

TEST(EntityLinkageTypeStringTest, ToStringNone) {
  EXPECT_EQ(entityLinkageTypeToString(EntityLinkageType::None), "None");
}

TEST(EntityLinkageTypeStringTest, ToStringInternal) {
  EXPECT_EQ(entityLinkageTypeToString(EntityLinkageType::Internal), "Internal");
}

TEST(EntityLinkageTypeStringTest, ToStringExternal) {
  EXPECT_EQ(entityLinkageTypeToString(EntityLinkageType::External), "External");
}

TEST(EntityLinkageTypeStringTest, FromStringNone) {
  EXPECT_EQ(entityLinkageTypeFromString("None"), EntityLinkageType::None);
}

TEST(EntityLinkageTypeStringTest, FromStringInternal) {
  EXPECT_EQ(entityLinkageTypeFromString("Internal"),
            EntityLinkageType::Internal);
}

TEST(EntityLinkageTypeStringTest, FromStringExternal) {
  EXPECT_EQ(entityLinkageTypeFromString("External"),
            EntityLinkageType::External);
}

TEST(EntityLinkageTypeStringTest, FromStringUnknown) {
  EXPECT_EQ(entityLinkageTypeFromString("none"), std::nullopt);
  EXPECT_EQ(entityLinkageTypeFromString("internal"), std::nullopt);
  EXPECT_EQ(entityLinkageTypeFromString("external"), std::nullopt);
  EXPECT_EQ(entityLinkageTypeFromString(""), std::nullopt);
  EXPECT_EQ(entityLinkageTypeFromString("unknown"), std::nullopt);
}

TEST(EntityLinkageTypeStringTest, RoundTrip) {
  for (auto LT : {EntityLinkageType::None, EntityLinkageType::Internal,
                  EntityLinkageType::External}) {
    EXPECT_EQ(entityLinkageTypeFromString(entityLinkageTypeToString(LT)), LT);
  }
}

} // namespace
