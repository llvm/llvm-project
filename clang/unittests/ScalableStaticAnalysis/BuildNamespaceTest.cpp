//===- BuildNamespaceTest.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysis/Core/Model/BuildNamespace.h"
#include "clang/ScalableStaticAnalysis/Core/Support/FormatProviders.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

namespace clang::ssaf {
namespace {

TEST(BuildNamespaceTest, DefaultConstructionIsEmpty) {
  BuildNamespace BN;
  EXPECT_TRUE(BN.empty());
}

TEST(BuildNamespaceTest, SingleNameConstruction) {
  BuildNamespace BN("test.cpp");
  EXPECT_FALSE(BN.empty());
}

TEST(BuildNamespaceTest, VectorConstruction) {
  BuildNamespace BN(std::vector<std::string>{"a", "b", "c"});
  EXPECT_FALSE(BN.empty());
}

TEST(BuildNamespaceTest, EqualityByNames) {
  BuildNamespace BN1("test.cpp");
  BuildNamespace BN2("test.cpp");
  BuildNamespace BN3("other.cpp");

  EXPECT_EQ(BN1, BN2);
  EXPECT_NE(BN1, BN3);
}

TEST(BuildNamespaceTest, EqualityAcrossLevels) {
  BuildNamespace Single("test.cpp");
  BuildNamespace Multi(std::vector<std::string>{"test.cpp", "app"});
  EXPECT_NE(Single, Multi);
}

TEST(BuildNamespaceTest, MakeQualifiedAppendsLevels) {
  BuildNamespace CU("test.cpp");
  BuildNamespace LU("app");
  auto Qualified = CU.makeQualified(LU);

  EXPECT_NE(Qualified, CU);
  EXPECT_NE(Qualified, LU);
}

TEST(BuildNamespaceTest, MakeQualifiedFromEmpty) {
  BuildNamespace Empty;
  BuildNamespace Named("test.cpp");
  auto Qualified = Empty.makeQualified(Named);
  EXPECT_EQ(Qualified, Named);
}

TEST(BuildNamespaceTest, MakeQualifiedWithEmpty) {
  BuildNamespace Named("test.cpp");
  BuildNamespace Empty;
  auto Qualified = Named.makeQualified(Empty);
  EXPECT_EQ(Qualified, Named);
}

TEST(BuildNamespaceTest, StreamOutputEmpty) {
  BuildNamespace BN;
  std::string S;
  llvm::raw_string_ostream(S) << BN;
  EXPECT_EQ(S, "BuildNamespace([])");
}

TEST(BuildNamespaceTest, StreamOutputSingle) {
  BuildNamespace BN("test.cpp");
  std::string S;
  llvm::raw_string_ostream(S) << BN;
  EXPECT_EQ(S, "BuildNamespace([test.cpp])");
}

TEST(BuildNamespaceTest, StreamOutputMultiple) {
  BuildNamespace BN =
      BuildNamespace("test.cpp").makeQualified(BuildNamespace("app"));
  std::string S;
  llvm::raw_string_ostream(S) << BN;
  EXPECT_EQ(S, "BuildNamespace([test.cpp, app])");
}

TEST(BuildNamespaceTest, FormatProviderEmpty) {
  EXPECT_EQ(llvm::formatv("{0}", BuildNamespace()).str(), "BuildNamespace([])");
}

TEST(BuildNamespaceTest, FormatProviderSingle) {
  EXPECT_EQ(llvm::formatv("{0}", BuildNamespace("test.cpp")).str(),
            "BuildNamespace([test.cpp])");
}

} // namespace
} // namespace clang::ssaf
