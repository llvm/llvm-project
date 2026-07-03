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

TEST(BuildNamespaceTest, Equality) {
  BuildNamespace BN1("test.cpp");
  BuildNamespace BN2("test.cpp");
  BuildNamespace BN3("other.cpp");

  EXPECT_EQ(BN1, BN2);
  EXPECT_NE(BN1, BN3);
}

// NestedBuildNamespace Tests

TEST(NestedBuildNamespaceTest, DefaultConstruction) {
  NestedBuildNamespace NBN;
  EXPECT_TRUE(NBN.empty());
}

TEST(NestedBuildNamespaceTest, SingleNamespaceConstruction) {
  BuildNamespace BN("test.cpp");
  NestedBuildNamespace NBN(BN);

  EXPECT_FALSE(NBN.empty());
}

TEST(NestedBuildNamespaceTest, Equality) {
  NestedBuildNamespace NBN1(BuildNamespace("test.cpp"));
  NestedBuildNamespace NBN2(BuildNamespace("test.cpp"));
  NestedBuildNamespace NBN3(BuildNamespace("other.cpp"));

  EXPECT_EQ(NBN1, NBN2);
  EXPECT_NE(NBN1, NBN3);
}

TEST(NestedBuildNamespaceTest, MakeQualified) {
  NestedBuildNamespace NBN1(BuildNamespace("test.cpp"));
  BuildNamespace LinkNS("app");
  NestedBuildNamespace NBN2(LinkNS);

  auto Qualified = NBN1.makeQualified(NBN2);

  EXPECT_NE(Qualified, NBN1);
  EXPECT_NE(Qualified, NBN2);
}

TEST(NestedBuildNamespaceTest, EmptyQualified) {
  NestedBuildNamespace Empty;
  NestedBuildNamespace NBN(BuildNamespace("test.cpp"));

  auto Qualified = Empty.makeQualified(NBN);
  EXPECT_EQ(Qualified, NBN);
}

TEST(BuildNamespaceTest, FormatProvider) {
  EXPECT_EQ(llvm::formatv("{0}", BuildNamespace("test.cpp")).str(),
            "BuildNamespace(test.cpp)");
}

TEST(NestedBuildNamespaceTest, FormatProvider) {
  NestedBuildNamespace NBN(BuildNamespace("test.cpp"));
  EXPECT_EQ(llvm::formatv("{0}", NBN).str(),
            "NestedBuildNamespace([BuildNamespace(test.cpp)])");
}

TEST(BuildNamespaceTest, StreamOutput) {
  BuildNamespace BN("test.cpp");
  std::string S;
  llvm::raw_string_ostream(S) << BN;
  EXPECT_EQ(S, "BuildNamespace(test.cpp)");
}

TEST(NestedBuildNamespaceTest, StreamOutputEmpty) {
  NestedBuildNamespace NBN;
  std::string S;
  llvm::raw_string_ostream(S) << NBN;
  EXPECT_EQ(S, "NestedBuildNamespace([])");
}

TEST(NestedBuildNamespaceTest, StreamOutputSingle) {
  NestedBuildNamespace NBN(BuildNamespace("test.cpp"));
  std::string S;
  llvm::raw_string_ostream(S) << NBN;
  EXPECT_EQ(S, "NestedBuildNamespace([BuildNamespace(test.cpp)])");
}

TEST(NestedBuildNamespaceTest, StreamOutputMultiple) {
  NestedBuildNamespace NBN(BuildNamespace("test.cpp"));
  NBN = NBN.makeQualified(NestedBuildNamespace(BuildNamespace("app")));
  std::string S;
  llvm::raw_string_ostream(S) << NBN;
  EXPECT_EQ(S, "NestedBuildNamespace([BuildNamespace(test.cpp), "
               "BuildNamespace(app)])");
}

} // namespace
} // namespace clang::ssaf
