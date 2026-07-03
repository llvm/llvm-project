//===- EntityNameTest.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysis/Core/Model/EntityName.h"
#include "clang/ScalableStaticAnalysis/Core/Model/BuildNamespace.h"
#include "clang/ScalableStaticAnalysis/Core/Support/FormatProviders.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

namespace clang::ssaf {
namespace {

TEST(EntityNameTest, Equality) {
  NestedBuildNamespace NBN1(BuildNamespace("test.cpp"));
  NestedBuildNamespace NBN2(BuildNamespace("test.cpp"));

  EntityName EN1("c:@F@foo", "", NBN1);
  EntityName EN2("c:@F@foo", "", NBN2);
  EntityName EN3("c:@F@bar", "", NBN1);

  EXPECT_EQ(EN1, EN2);
  EXPECT_NE(EN1, EN3);
}

TEST(EntityNameTest, EqualityWithDifferentSuffix) {
  NestedBuildNamespace NBN(BuildNamespace("test.cpp"));

  EntityName EN1("c:@F@foo", "1", NBN);
  EntityName EN2("c:@F@foo", "2", NBN);

  EXPECT_NE(EN1, EN2);
}

TEST(EntityNameTest, EqualityWithDifferentNamespace) {
  NestedBuildNamespace NBN1(BuildNamespace("test1.cpp"));
  NestedBuildNamespace NBN2(BuildNamespace("test2.cpp"));

  EntityName EN1("c:@F@foo", "", NBN1);
  EntityName EN2("c:@F@foo", "", NBN2);

  EXPECT_NE(EN1, EN2);
}

TEST(EntityNameTest, MakeQualified) {
  NestedBuildNamespace NBN1(BuildNamespace("test.cpp"));
  EntityName EN("c:@F@foo", "", NBN1);

  NestedBuildNamespace NBN2(BuildNamespace("app"));

  auto Qualified = EN.makeQualified(NBN2);

  EXPECT_NE(Qualified, EN);
}

TEST(EntityNameTest, FormatProvider) {
  NestedBuildNamespace NBN(BuildNamespace("test.cpp"));
  EntityName EN("c:@F@foo", "", NBN);
  EXPECT_EQ(llvm::formatv("{0}", EN).str(),
            "EntityName(c:@F@foo, , "
            "NestedBuildNamespace([BuildNamespace(test.cpp)]))");
}

TEST(EntityNameTest, StreamOutputNoSuffix) {
  NestedBuildNamespace NBN(BuildNamespace("test.cpp"));
  EntityName EN("c:@F@foo", "", NBN);
  std::string S;
  llvm::raw_string_ostream(S) << EN;
  EXPECT_EQ(S, "EntityName(c:@F@foo, , "
               "NestedBuildNamespace([BuildNamespace(test.cpp)]))");
}

TEST(EntityNameTest, StreamOutputWithSuffix) {
  NestedBuildNamespace NBN(BuildNamespace("test.cpp"));
  EntityName EN("c:@F@foo", "1", NBN);
  std::string S;
  llvm::raw_string_ostream(S) << EN;
  EXPECT_EQ(S, "EntityName(c:@F@foo, 1, "
               "NestedBuildNamespace([BuildNamespace(test.cpp)]))");
}

} // namespace
} // namespace clang::ssaf
