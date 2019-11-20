//===-- ASTTests.cpp --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AST.h"

#include "Annotations.h"
#include "TestTU.h"
#include "clang/Basic/SourceManager.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

TEST(ShortenNamespace, All) {
  ASSERT_EQ("TestClass", shortenNamespace("TestClass", ""));

  ASSERT_EQ("TestClass", shortenNamespace(
      "testnamespace::TestClass", "testnamespace"));

  ASSERT_EQ(
      "namespace1::TestClass",
      shortenNamespace("namespace1::TestClass", "namespace2"));

  ASSERT_EQ("TestClass",
            shortenNamespace("testns1::testns2::TestClass",
                             "testns1::testns2"));

  ASSERT_EQ(
      "testns2::TestClass",
      shortenNamespace("testns1::testns2::TestClass", "testns1"));

  ASSERT_EQ("TestClass<testns1::OtherClass>",
            shortenNamespace(
                "testns1::TestClass<testns1::OtherClass>", "testns1"));
}

TEST(GetDeducedType, KwAutoExpansion) {
  struct Test {
    StringRef AnnotatedCode;
    const char *DeducedType;
  } Tests[] = {
      {"^auto i = 0;", "int"},
      {"^auto f(){ return 1;};", "int"},
  };
  for (Test T : Tests) {
    Annotations File(T.AnnotatedCode);
    auto AST = TestTU::withCode(File.code()).build();
    ASSERT_TRUE(AST.getDiagnostics().empty())
        << AST.getDiagnostics().begin()->Message;
    SourceManagerForFile SM("foo.cpp", File.code());

    for (Position Pos : File.points()) {
      auto Location = sourceLocationInMainFile(SM.get(), Pos);
      ASSERT_TRUE(!!Location) << llvm::toString(Location.takeError());
      auto DeducedType = getDeducedType(AST.getASTContext(), *Location);
      EXPECT_EQ(DeducedType->getAsString(), T.DeducedType);
    }
  }
}

} // namespace
} // namespace clangd
} // namespace clang
