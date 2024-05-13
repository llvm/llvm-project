//===-- ExpandDeducedTypeTests.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TweakTesting.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::StartsWith;

namespace clang {
namespace clangd {
namespace {

TWEAK_TEST(ExpandDeducedType);

TEST_F(ExpandDeducedTypeTest, Test) {
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
              StartsWith("fail: Could not expand type"));
  // function references should not be replaced
  EXPECT_THAT(apply("au^to &x = ns::Func;"),
              StartsWith("fail: Could not expand type"));
  // lambda types are not replaced
  EXPECT_UNAVAILABLE("au^to x = []{};");
  // inline namespaces
  EXPECT_EQ(apply("au^to x = inl_ns::Visible();"),
            "inl_ns::Visible x = inl_ns::Visible();");
  // local class
  EXPECT_EQ(apply("namespace x { void y() { struct S{}; ^auto z = S(); } }"),
            "namespace x { void y() { struct S{}; S z = S(); } }");
  // replace pointers
  EXPECT_EQ(apply(R"cpp(au^to x = "test";)cpp"),
            R"cpp(const char * x = "test";)cpp");
  // pointers to an array are not replaced
  EXPECT_THAT(apply(R"cpp(au^to s = &"foobar";)cpp"),
              StartsWith("fail: Could not expand type"));

  EXPECT_EQ(apply("ns::Class * foo() { au^to c = foo(); }"),
            "ns::Class * foo() { ns::Class * c = foo(); }");
  EXPECT_EQ(
      apply("void ns::Func() { au^to x = new ns::Class::Nested{}; }"),
      "void ns::Func() { ns::Class::Nested * x = new ns::Class::Nested{}; }");

  EXPECT_EQ(apply("dec^ltype(auto) x = 10;"), "int x = 10;");
  EXPECT_EQ(apply("decltype(au^to) x = 10;"), "int x = 10;");
  // references to array types are not replaced
  EXPECT_THAT(apply(R"cpp(decl^type(auto) s = "foobar"; // error-ok)cpp"),
              StartsWith("fail: Could not expand type"));
  // array types are not replaced
  EXPECT_THAT(apply("int arr[10]; decl^type(auto) foobar = arr; // error-ok"),
              StartsWith("fail: Could not expand type"));
  // pointers to an array are not replaced
  EXPECT_THAT(apply(R"cpp(decl^type(auto) s = &"foobar";)cpp"),
              StartsWith("fail: Could not expand type"));
  // expanding types in structured bindings is syntactically invalid.
  EXPECT_UNAVAILABLE("const ^auto &[x,y] = (int[]){1,2};");

  // unknown types in a template should not be replaced
  EXPECT_THAT(apply("template <typename T> void x() { ^auto y = T::z(); }"),
              StartsWith("fail: Could not deduce type for 'auto' type"));

  // check primitive type
  EXPECT_EQ(apply("decl^type(0) i;"), "int i;");
  // function should not be replaced
  EXPECT_THAT(apply("void f(); decl^type(f) g;"),
              StartsWith("fail: Could not expand type"));
  // check return type in function proto
  EXPECT_EQ(apply("decl^type(0) f();"), "int f();");
  // check trailing return type
  EXPECT_EQ(apply("auto f() -> decl^type(0) { return 0; }"),
            "auto f() -> int { return 0; }");
  // check function parameter type
  EXPECT_EQ(apply("void f(decl^type(0));"), "void f(int);");
  // check template parameter type
  EXPECT_EQ(apply("template <decl^type(0)> struct Foobar {};"),
            "template <int> struct Foobar {};");
  // check default template argument
  EXPECT_EQ(apply("template <class = decl^type(0)> class Foo {};"),
            "template <class = int> class Foo {};");
  // check template argument
  EXPECT_EQ(apply("template <class> class Bar {}; Bar<decl^type(0)> b;"),
            "template <class> class Bar {}; Bar<int> b;");
  // dependent types are not replaced
  EXPECT_THAT(apply("template <class T> struct Foobar { decl^type(T{}) t; };"),
              StartsWith("fail: Could not expand a dependent type"));
  // references to array types are not replaced
  EXPECT_THAT(apply(R"cpp(decl^type("foobar") s; // error-ok)cpp"),
              StartsWith("fail: Could not expand type"));
  // array types are not replaced
  EXPECT_THAT(apply("int arr[10]; decl^type(arr) foobar;"),
              StartsWith("fail: Could not expand type"));
  // pointers to an array are not replaced
  EXPECT_THAT(apply(R"cpp(decl^type(&"foobar") s;)cpp"),
              StartsWith("fail: Could not expand type"));

  ExtraArgs.push_back("-std=c++20");
  EXPECT_UNAVAILABLE("template <au^to X> class Y;");

  EXPECT_THAT(apply("auto X = [](^auto){};"),
              StartsWith("fail: Could not deduce"));
  EXPECT_EQ(apply("auto X = [](^auto){return 0;}; int Y = X(42);"),
            "auto X = [](int){return 0;}; int Y = X(42);");
  EXPECT_THAT(apply("auto X = [](^auto){return 0;}; int Y = X(42) + X('c');"),
              StartsWith("fail: Could not deduce"));
  // FIXME: should work on constrained auto params, once SourceRange is fixed.
  EXPECT_UNAVAILABLE("template<class> concept C = true;"
                     "auto X = [](C ^auto *){return 0;};");

  // lambda should not be replaced
  EXPECT_UNAVAILABLE("auto f = [](){}; decl^type(f) g;");
  EXPECT_UNAVAILABLE("decl^type([]{}) f;");
}

} // namespace
} // namespace clangd
} // namespace clang
