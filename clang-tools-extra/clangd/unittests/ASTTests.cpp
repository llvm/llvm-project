//===-- ASTTests.cpp --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AST.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

TEST(ExpandAutoType, ShortenNamespace) {
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


} // namespace
} // namespace clangd
} // namespace clang
