//===- unittests/Analysis/Scalable/BuildNamespaceTest.cpp ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/Model/BuildNamespace.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

namespace clang::ssaf {
namespace {

TEST(BuildNamespaceTest, Equality) {
  auto BN1 = BuildNamespace::makeCompilationUnit("test.cpp");
  auto BN2 = BuildNamespace::makeCompilationUnit("test.cpp");
  auto BN3 = BuildNamespace::makeCompilationUnit("other.cpp");

  EXPECT_EQ(BN1, BN2);
  EXPECT_NE(BN1, BN3);
}

TEST(BuildNamespaceTest, DifferentKinds) {
  BuildNamespace CU(BuildNamespaceKind::CompilationUnit, "test");
  BuildNamespace LU(BuildNamespaceKind::LinkUnit, "test");

  EXPECT_NE(CU, LU);
}

TEST(BuildNamespaceTest, ToStringRoundtripCompilationUnit) {
  auto Kind = BuildNamespaceKind::CompilationUnit;
  auto Str = toString(Kind);
  auto Parsed = parseBuildNamespaceKind(Str);

  ASSERT_TRUE(Parsed.has_value());
  EXPECT_EQ(Kind, *Parsed);
}

TEST(BuildNamespaceTest, ToStringRoundtripLinkUnit) {
  auto Kind = BuildNamespaceKind::LinkUnit;
  auto Str = toString(Kind);
  auto Parsed = parseBuildNamespaceKind(Str);

  ASSERT_TRUE(Parsed.has_value());
  EXPECT_EQ(Kind, *Parsed);
}

// NestedBuildNamespace Tests

TEST(NestedBuildNamespaceTest, DefaultConstruction) {
  NestedBuildNamespace NBN;
  EXPECT_TRUE(NBN.empty());
}

TEST(NestedBuildNamespaceTest, SingleNamespaceConstruction) {
  auto BN = BuildNamespace::makeCompilationUnit("test.cpp");
  NestedBuildNamespace NBN(BN);

  EXPECT_FALSE(NBN.empty());
}

TEST(NestedBuildNamespaceTest, MakeTU) {
  auto NBN = NestedBuildNamespace::makeCompilationUnit("test.cpp");
  EXPECT_FALSE(NBN.empty());
}

TEST(NestedBuildNamespaceTest, Equality) {
  auto NBN1 = NestedBuildNamespace::makeCompilationUnit("test.cpp");
  auto NBN2 = NestedBuildNamespace::makeCompilationUnit("test.cpp");
  auto NBN3 = NestedBuildNamespace::makeCompilationUnit("other.cpp");

  EXPECT_EQ(NBN1, NBN2);
  EXPECT_NE(NBN1, NBN3);
}

TEST(NestedBuildNamespaceTest, MakeQualified) {
  auto NBN1 = NestedBuildNamespace::makeCompilationUnit("test.cpp");
  BuildNamespace LinkNS(BuildNamespaceKind::LinkUnit, "app");
  NestedBuildNamespace NBN2(LinkNS);

  auto Qualified = NBN1.makeQualified(NBN2);

  EXPECT_NE(Qualified, NBN1);
  EXPECT_NE(Qualified, NBN2);
}

TEST(NestedBuildNamespaceTest, EmptyQualified) {
  NestedBuildNamespace Empty;
  auto NBN = NestedBuildNamespace::makeCompilationUnit("test.cpp");

  auto Qualified = Empty.makeQualified(NBN);
  EXPECT_EQ(Qualified, NBN);
}

TEST(BuildNamespaceTest, StreamOutputCompilationUnit) {
  BuildNamespace BN(BuildNamespaceKind::CompilationUnit, "test.cpp");
  std::string S;
  llvm::raw_string_ostream(S) << BN;
  EXPECT_EQ(S, "BuildNamespace(compilation_unit, test.cpp)");
}

TEST(BuildNamespaceTest, StreamOutputLinkUnit) {
  BuildNamespace BN(BuildNamespaceKind::LinkUnit, "app");
  std::string S;
  llvm::raw_string_ostream(S) << BN;
  EXPECT_EQ(S, "BuildNamespace(link_unit, app)");
}

TEST(NestedBuildNamespaceTest, StreamOutputEmpty) {
  NestedBuildNamespace NBN;
  std::string S;
  llvm::raw_string_ostream(S) << NBN;
  EXPECT_EQ(S, "NestedBuildNamespace([])");
}

TEST(NestedBuildNamespaceTest, StreamOutputSingle) {
  NestedBuildNamespace NBN(
      BuildNamespace(BuildNamespaceKind::CompilationUnit, "test.cpp"));
  std::string S;
  llvm::raw_string_ostream(S) << NBN;
  EXPECT_EQ(
      S, "NestedBuildNamespace([BuildNamespace(compilation_unit, test.cpp)])");
}

TEST(NestedBuildNamespaceTest, StreamOutputMultiple) {
  NestedBuildNamespace NBN(
      BuildNamespace(BuildNamespaceKind::CompilationUnit, "test.cpp"));
  NBN = NBN.makeQualified(NestedBuildNamespace(
      BuildNamespace(BuildNamespaceKind::LinkUnit, "app")));
  std::string S;
  llvm::raw_string_ostream(S) << NBN;
  EXPECT_EQ(S, "NestedBuildNamespace([BuildNamespace(compilation_unit, "
               "test.cpp), BuildNamespace(link_unit, app)])");
}

} // namespace
} // namespace clang::ssaf
