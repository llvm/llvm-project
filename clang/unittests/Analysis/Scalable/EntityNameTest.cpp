//===- unittests/Analysis/Scalable/EntityNameTest.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/Model/EntityName.h"
#include "clang/Analysis/Scalable/Model/BuildNamespace.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

namespace clang::ssaf {
namespace {

TEST(EntityNameTest, Equality) {
  auto NBN1 = NestedBuildNamespace::makeCompilationUnit("test.cpp");
  auto NBN2 = NestedBuildNamespace::makeCompilationUnit("test.cpp");

  EntityName EN1("c:@F@foo", "", NBN1);
  EntityName EN2("c:@F@foo", "", NBN2);
  EntityName EN3("c:@F@bar", "", NBN1);

  EXPECT_EQ(EN1, EN2);
  EXPECT_NE(EN1, EN3);
}

TEST(EntityNameTest, EqualityWithDifferentSuffix) {
  auto NBN = NestedBuildNamespace::makeCompilationUnit("test.cpp");

  EntityName EN1("c:@F@foo", "1", NBN);
  EntityName EN2("c:@F@foo", "2", NBN);

  EXPECT_NE(EN1, EN2);
}

TEST(EntityNameTest, EqualityWithDifferentNamespace) {
  auto NBN1 = NestedBuildNamespace::makeCompilationUnit("test1.cpp");
  auto NBN2 = NestedBuildNamespace::makeCompilationUnit("test2.cpp");

  EntityName EN1("c:@F@foo", "", NBN1);
  EntityName EN2("c:@F@foo", "", NBN2);

  EXPECT_NE(EN1, EN2);
}

TEST(EntityNameTest, MakeQualified) {
  auto NBN1 = NestedBuildNamespace::makeCompilationUnit("test.cpp");
  EntityName EN("c:@F@foo", "", NBN1);

  BuildNamespace LinkNS(BuildNamespaceKind::LinkUnit, "app");
  NestedBuildNamespace NBN2(LinkNS);

  auto Qualified = EN.makeQualified(NBN2);

  EXPECT_NE(Qualified, EN);
}

TEST(EntityNameTest, StreamOutputNoSuffix) {
  NestedBuildNamespace NBN(
      BuildNamespace(BuildNamespaceKind::CompilationUnit, "test.cpp"));
  EntityName EN("c:@F@foo", "", NBN);
  std::string S;
  llvm::raw_string_ostream(S) << EN;
  EXPECT_EQ(
      S, "EntityName(c:@F@foo, , "
         "NestedBuildNamespace([BuildNamespace(compilation_unit, test.cpp)]))");
}

TEST(EntityNameTest, StreamOutputWithSuffix) {
  NestedBuildNamespace NBN(
      BuildNamespace(BuildNamespaceKind::CompilationUnit, "test.cpp"));
  EntityName EN("c:@F@foo", "1", NBN);
  std::string S;
  llvm::raw_string_ostream(S) << EN;
  EXPECT_EQ(
      S, "EntityName(c:@F@foo, 1, "
         "NestedBuildNamespace([BuildNamespace(compilation_unit, test.cpp)]))");
}

} // namespace
} // namespace clang::ssaf
