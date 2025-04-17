//===-- SwapBinaryOperandsTests.cpp -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TweakTesting.h"
#include "gmock/gmock-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

TWEAK_TEST(SwapBinaryOperands);

TEST_F(SwapBinaryOperandsTest, Test) {
  Context = Function;
  EXPECT_EQ(apply("int *p = nullptr; bool c = ^p == nullptr;"),
            "int *p = nullptr; bool c = nullptr == p;");
  EXPECT_EQ(apply("int *p = nullptr; bool c = p ^== nullptr;"),
            "int *p = nullptr; bool c = nullptr == p;");
  EXPECT_EQ(apply("int x = 3; bool c = ^x >= 5;"),
            "int x = 3; bool c = 5 <= x;");
  EXPECT_EQ(apply("int x = 3; bool c = x >^= 5;"),
            "int x = 3; bool c = 5 <= x;");
  EXPECT_EQ(apply("int x = 3; bool c = x >=^ 5;"),
            "int x = 3; bool c = 5 <= x;");
  EXPECT_EQ(apply("int x = 3; bool c = x >=^ 5;"),
            "int x = 3; bool c = 5 <= x;");
  EXPECT_EQ(apply("int f(); int x = 3; bool c = x >=^ f();"),
            "int f(); int x = 3; bool c = f() <= x;");
  EXPECT_EQ(apply(R"cpp(
            int f();
            #define F f
            int x = 3; bool c = x >=^ F();
            )cpp"),
            R"cpp(
            int f();
            #define F f
            int x = 3; bool c = F() <= x;
            )cpp");
  EXPECT_EQ(apply(R"cpp(
            int f();
            #define F f()
            int x = 3; bool c = x >=^ F;
            )cpp"),
            R"cpp(
            int f();
            #define F f()
            int x = 3; bool c = F <= x;
            )cpp");
  EXPECT_EQ(apply(R"cpp(
            int f(bool);
            #define F(v) f(v)
            int x = 0;
            bool c = F(x^ < 5);
            )cpp"),
            R"cpp(
            int f(bool);
            #define F(v) f(v)
            int x = 0;
            bool c = F(5 > x);
            )cpp");
  ExtraArgs = {"-std=c++20"};
  Context = CodeContext::File;
  EXPECT_UNAVAILABLE(R"cpp(
            namespace std {
                struct strong_ordering {
                    int val; 
                    static const strong_ordering less;
                    static const strong_ordering equivalent;
                    static const strong_ordering equal;
                    static const strong_ordering greater;
                }; 
                    inline constexpr strong_ordering strong_ordering::less {-1};
                    inline constexpr strong_ordering strong_ordering::equivalent {0};
                    inline constexpr strong_ordering strong_ordering::equal {0};
                    inline constexpr strong_ordering strong_ordering::greater {1};
            };
            #define F(v) v
            int x = 0;
            auto c = F(5^ <=> x);
            )cpp");
}

} // namespace
} // namespace clangd
} // namespace clang
