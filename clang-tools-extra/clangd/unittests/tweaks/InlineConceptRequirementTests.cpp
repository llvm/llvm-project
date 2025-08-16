//===-- InlineConceptRequirementTests.cpp -----------------------*- C++ -*-===//
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

TWEAK_TEST(InlineConceptRequirement);

TEST_F(InlineConceptRequirementTest, Test) {
  Header = R"cpp(
      template <typename T>
      concept foo = true;

      template <typename T>
      concept bar = true;

      template <typename T, typename U>
      concept baz = true;
    )cpp";

  ExtraArgs = {"-std=c++20"};

  //
  // Extra spaces are expected and will be stripped by the formatter.
  //

  EXPECT_EQ(
      apply("template <typename T, typename U> void f(T) requires f^oo<U> {}"),
      "template <typename T, foo U> void f(T)   {}");

  EXPECT_EQ(
      apply("template <typename T, typename U> requires foo<^T> void f(T) {}"),
      "template <foo T, typename U>   void f(T) {}");

  EXPECT_EQ(apply("template <template <typename> class FooBar, typename T> "
                  "void f() requires foo<^T> {}"),
            "template <template <typename> class FooBar, foo T> void f()   {}");

  EXPECT_AVAILABLE(R"cpp(
      template <typename T> void f(T)
        requires ^f^o^o^<^T^> {}
    )cpp");

  EXPECT_AVAILABLE(R"cpp(
      template <typename T> requires ^f^o^o^<^T^>
      void f(T) {}
    )cpp");

  EXPECT_AVAILABLE(R"cpp(
      template <typename T, typename U> void f(T)
        requires ^f^o^o^<^T^> {}
    )cpp");

  EXPECT_AVAILABLE(R"cpp(
      template <template <typename> class FooBar, typename T>
      void foobar() requires ^f^o^o^<^T^>
      {}
    )cpp");

  EXPECT_UNAVAILABLE(R"cpp(
      template <bar T> void f(T)
        requires ^f^o^o^<^T^> {}
    )cpp");

  EXPECT_UNAVAILABLE(R"cpp(
      template <typename T, typename U> void f(T, U)
        requires ^b^a^z^<^T^,^ ^U^> {}
    )cpp");

  EXPECT_UNAVAILABLE(R"cpp(
      template <typename T> void f(T)
        requires ^f^o^o^<^T^>^ ^&^&^ ^b^a^r^<^T^> {}
    )cpp");

  EXPECT_UNAVAILABLE(R"cpp(
      template <typename T>
      concept ^f^o^o^b^a^r = requires(^T^ ^x^) {
        {x} -> ^f^o^o^;
      };
    )cpp");
}

} // namespace
} // namespace clangd
} // namespace clang
