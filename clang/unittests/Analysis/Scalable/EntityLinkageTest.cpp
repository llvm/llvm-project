//===- unittests/Analysis/Scalable/EntityLinkageTest.cpp -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/Model/EntityLinkage.h"
#include "gtest/gtest.h"

namespace clang::ssaf {

namespace {

TEST(EntityLinkageTest, GetLinkageReturnsCorrectValue) {
  EntityLinkage Linkage;
  EntityLinkage NoneLinkage(EntityLinkage::LinkageType::None);
  EntityLinkage InternalLinkage(EntityLinkage::LinkageType::Internal);
  EntityLinkage ExternalLinkage(EntityLinkage::LinkageType::External);

  EXPECT_EQ(Linkage.getLinkage(), EntityLinkage::LinkageType::None);
  EXPECT_EQ(NoneLinkage.getLinkage(), EntityLinkage::LinkageType::None);
  EXPECT_EQ(InternalLinkage.getLinkage(), EntityLinkage::LinkageType::Internal);
  EXPECT_EQ(ExternalLinkage.getLinkage(), EntityLinkage::LinkageType::External);
}

TEST(EntityLinkageTest, CopyConstructor) {
  EntityLinkage Original(EntityLinkage::LinkageType::External);
  EntityLinkage Copy = Original;

  EXPECT_EQ(Copy.getLinkage(), EntityLinkage::LinkageType::External);
  EXPECT_EQ(Copy.getLinkage(), Original.getLinkage());
}

TEST(EntityLinkageTest, AssignmentOperator) {
  EntityLinkage Linkage1(EntityLinkage::LinkageType::None);
  EntityLinkage Linkage2(EntityLinkage::LinkageType::External);

  Linkage1 = Linkage2;

  EXPECT_EQ(Linkage1.getLinkage(), EntityLinkage::LinkageType::External);
  EXPECT_EQ(Linkage1.getLinkage(), Linkage2.getLinkage());
}

} // namespace

} // namespace clang::ssaf
