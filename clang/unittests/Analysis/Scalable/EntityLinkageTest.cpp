//===- unittests/Analysis/Scalable/EntityLinkageTest.cpp -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/Model/EntityLinkage.h"
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

} // namespace
