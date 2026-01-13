//===-- AbbreviateFunctionTemplateTests.cpp ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TweakTesting.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

TWEAK_TEST(AbbreviateFunctionTemplate);

TEST_F(AbbreviateFunctionTemplateTest, Test) {
  Header = R"cpp(
      template <typename T>
      concept foo = true;

      template <typename T>
      concept bar = true;

      template <typename T, typename U>
      concept baz = true;

      template <typename T>
      class list;
  )cpp";

  ExtraArgs = {"-std=c++20"};

  EXPECT_EQ(apply("template <typename T> auto ^fun(T param) {}"),
            " auto fun( auto param) {}");
  EXPECT_EQ(apply("template <foo T> auto ^fun(T param) {}"),
            " auto fun( foo auto param) {}");
  EXPECT_EQ(apply("template <foo T> auto ^fun(T) {}"),
            " auto fun( foo auto ) {}");
  EXPECT_EQ(apply("template <foo T> auto ^fun(T[]) {}"),
            " auto fun( foo auto  [ ]) {}");
  EXPECT_EQ(apply("template <foo T> auto ^fun(T const * param[]) {}"),
            " auto fun( foo auto const * param [ ]) {}");
  EXPECT_EQ(apply("template <baz<int> T> auto ^fun(T param) {}"),
            " auto fun( baz <int> auto param) {}");
  EXPECT_EQ(apply("template <foo T, bar U> auto ^fun(T param1, U param2) {}"),
            " auto fun( foo auto param1,  bar auto param2) {}");
  EXPECT_EQ(apply("template <foo T> auto ^fun(T const ** param) {}"),
            " auto fun( foo auto const * * param) {}");
  EXPECT_EQ(apply("template <typename...ArgTypes> auto ^fun(ArgTypes...params) "
                  "-> void{}"),
            " auto fun( auto ... params) -> void{}");

  EXPECT_AVAILABLE("temp^l^ate <type^name ^T> au^to fu^n^(^T par^am) {}");
  EXPECT_AVAILABLE("t^emplat^e <fo^o ^T> aut^o fu^n^(^T ^para^m) -> void {}");
  EXPECT_AVAILABLE(
      "^templa^te <f^oo T^> a^uto ^fun(^T const ** para^m) -> void {}");
  EXPECT_AVAILABLE("templa^te <type^name...ArgTypes> auto "
                   "fu^n(ArgTy^pes...^para^ms) -> void{}");

  EXPECT_UNAVAILABLE(
      "templ^ate<typenam^e T> auto f^u^n(list<T> pa^ram) -> void {}");

  // Template parameters need to be in the same order as the function parameters
  EXPECT_UNAVAILABLE(
      "tem^plate<type^name ^T, typen^ame ^U> auto f^un(^U, ^T) -> void {}");

  // Template parameter type can't be used within the function body
  EXPECT_UNAVAILABLE("templ^ate<cl^ass T>"
                     "aut^o fu^n(T param) -> v^oid { T bar; }");
}

} // namespace
} // namespace clangd
} // namespace clang
