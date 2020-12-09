//===-- AnnotateHighlightingsTests.cpp --------------------------*- C++ -*-===//
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

TWEAK_TEST(AnnotateHighlightings);

TEST_F(AnnotateHighlightingsTest, Test) {
  EXPECT_AVAILABLE("^vo^id^ ^f(^) {^}^"); // available everywhere.
  EXPECT_AVAILABLE("[[int a; int b;]]");
  EXPECT_EQ("void /* entity.name.function.cpp */f() {}", apply("void ^f() {}"));

  EXPECT_EQ(apply("[[void f1(); void f2();]]"),
            "void /* entity.name.function.cpp */f1(); "
            "void /* entity.name.function.cpp */f2();");

  EXPECT_EQ(apply("void f1(); void f2() {^}"),
            "void f1(); "
            "void /* entity.name.function.cpp */f2() {}");
}

} // namespace
} // namespace clangd
} // namespace clang
