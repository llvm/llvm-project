//===-- ExpandAutoTypeTests.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestTU.h"
#include "TweakTesting.h"
#include "gmock/gmock-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::StartsWith;

namespace clang {
namespace clangd {
namespace {

TWEAK_TEST(ExpandAutoType);

TEST_F(ExpandAutoTypeTest, Test) {
  Header = R"cpp(
    namespace ns {
      struct Class {
        struct Nested {};
      };
      void Func();
    }
    inline namespace inl_ns {
      namespace {
        struct Visible {};
      }
    }
  )cpp";

  EXPECT_AVAILABLE("^a^u^t^o^ i = 0;");
  EXPECT_UNAVAILABLE("auto ^i^ ^=^ ^0^;^");

  // check primitive type
  EXPECT_EQ(apply("[[auto]] i = 0;"), "int i = 0;");
  EXPECT_EQ(apply("au^to i = 0;"), "int i = 0;");
  // check classes and namespaces
  EXPECT_EQ(apply("^auto C = ns::Class::Nested();"),
            "ns::Class::Nested C = ns::Class::Nested();");
  // check that namespaces are shortened
  EXPECT_EQ(apply("namespace ns { void f() { ^auto C = Class(); } }"),
            "namespace ns { void f() { Class C = Class(); } }");
  // undefined functions should not be replaced
  EXPECT_THAT(apply("au^to x = doesnt_exist(); // error-ok"),
              StartsWith("fail: Could not deduce type for 'auto' type"));
  // function pointers should not be replaced
  EXPECT_THAT(apply("au^to x = &ns::Func;"),
              StartsWith("fail: Could not expand type of function pointer"));
  // lambda types are not replaced
  EXPECT_UNAVAILABLE("au^to x = []{};");
  // inline namespaces
  EXPECT_EQ(apply("au^to x = inl_ns::Visible();"),
            "inl_ns::Visible x = inl_ns::Visible();");
  // local class
  EXPECT_EQ(apply("namespace x { void y() { struct S{}; ^auto z = S(); } }"),
            "namespace x { void y() { struct S{}; S z = S(); } }");
  // replace array types
  EXPECT_EQ(apply(R"cpp(au^to x = "test";)cpp"),
            R"cpp(const char * x = "test";)cpp");

  EXPECT_EQ(apply("ns::Class * foo() { au^to c = foo(); }"),
            "ns::Class * foo() { ns::Class * c = foo(); }");
  EXPECT_EQ(
      apply("void ns::Func() { au^to x = new ns::Class::Nested{}; }"),
      "void ns::Func() { ns::Class::Nested * x = new ns::Class::Nested{}; }");

  EXPECT_EQ(apply("dec^ltype(auto) x = 10;"), "int x = 10;");
  EXPECT_EQ(apply("decltype(au^to) x = 10;"), "int x = 10;");
  // expanding types in structured bindings is syntactically invalid.
  EXPECT_UNAVAILABLE("const ^auto &[x,y] = (int[]){1,2};");

  // unknown types in a template should not be replaced
  EXPECT_THAT(apply("template <typename T> void x() { ^auto y = T::z(); }"),
              StartsWith("fail: Could not deduce type for 'auto' type"));

  ExtraArgs.push_back("-std=c++17");
  EXPECT_UNAVAILABLE("template <au^to X> class Y;");
}

} // namespace
} // namespace clangd
} // namespace clang
