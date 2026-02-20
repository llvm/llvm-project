//===- unittests/Analysis/Scalable/EntityLinkageTest.cpp -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/Model/EntityLinkage.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using clang::ssaf::EntityLinkage;

namespace {

constexpr inline auto None = EntityLinkage::LinkageType::None;
constexpr inline auto Internal = EntityLinkage::LinkageType::Internal;
constexpr inline auto External = EntityLinkage::LinkageType::External;

TEST(EntityLinkageTest, Constructor) {
  EntityLinkage NoneLinkage(None);
  EntityLinkage InternalLinkage(Internal);
  EntityLinkage ExternalLinkage(External);

  EXPECT_EQ(NoneLinkage.getLinkage(), None);
  EXPECT_EQ(InternalLinkage.getLinkage(), Internal);
  EXPECT_EQ(ExternalLinkage.getLinkage(), External);
}

TEST(EntityLinkageTest, CopyConstructor) {
  EntityLinkage Original(External);
  EntityLinkage Copy = Original;

  EXPECT_EQ(Copy.getLinkage(), External);
  EXPECT_EQ(Copy.getLinkage(), Original.getLinkage());
}

TEST(EntityLinkageTest, AssignmentOperator) {
  EntityLinkage Linkage1(None);
  EntityLinkage Linkage2(External);

  Linkage1 = Linkage2;

  EXPECT_EQ(Linkage1.getLinkage(), External);
  EXPECT_EQ(Linkage1.getLinkage(), Linkage2.getLinkage());
}

TEST(EntityLinkageTest, EqualityOperatorReflexive) {
  EXPECT_TRUE(EntityLinkage(None) == EntityLinkage(None));
  EXPECT_TRUE(EntityLinkage(Internal) == EntityLinkage(Internal));
  EXPECT_TRUE(EntityLinkage(External) == EntityLinkage(External));
}

TEST(EntityLinkageTest, EqualityOperatorDistinct) {
  EXPECT_FALSE(EntityLinkage(None) == EntityLinkage(Internal));
  EXPECT_FALSE(EntityLinkage(None) == EntityLinkage(External));
  EXPECT_FALSE(EntityLinkage(Internal) == EntityLinkage(External));
}

TEST(EntityLinkageTest, InequalityOperatorDistinct) {
  EXPECT_TRUE(EntityLinkage(None) != EntityLinkage(Internal));
  EXPECT_TRUE(EntityLinkage(None) != EntityLinkage(External));
  EXPECT_TRUE(EntityLinkage(Internal) != EntityLinkage(External));
}

TEST(EntityLinkageTest, InequalityOperatorReflexive) {
  EXPECT_FALSE(EntityLinkage(None) != EntityLinkage(None));
  EXPECT_FALSE(EntityLinkage(Internal) != EntityLinkage(Internal));
  EXPECT_FALSE(EntityLinkage(External) != EntityLinkage(External));
}

TEST(EntityLinkageTest, StreamOutputNone) {
  std::string S;
  llvm::raw_string_ostream(S) << EntityLinkage(None);
  EXPECT_EQ(S, "EntityLinkage(None)");
}

TEST(EntityLinkageTest, StreamOutputInternal) {
  std::string S;
  llvm::raw_string_ostream(S) << EntityLinkage(Internal);
  EXPECT_EQ(S, "EntityLinkage(Internal)");
}

TEST(EntityLinkageTest, StreamOutputExternal) {
  std::string S;
  llvm::raw_string_ostream(S) << EntityLinkage(External);
  EXPECT_EQ(S, "EntityLinkage(External)");
}

} // namespace
