//===- LUSummaryTest.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Core/EntityLinker/LUSummary.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/BuildNamespace.h"
#include "gtest/gtest.h"

namespace clang::ssaf {
namespace {

TEST(LUSummaryTest, GetNamespace) {
  BuildNamespace LU(BuildNamespaceKind::LinkUnit, "app");
  NestedBuildNamespace NS(LU);
  LUSummary Summary(NS);

  EXPECT_EQ(Summary.getNamespace(), NS);
}

} // namespace
} // namespace clang::ssaf
