//===-- DefineOutline.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TweakTesting.h"
#include "gtest/gtest.h"

namespace clang::clangd {
namespace {

TWEAK_TEST(ScopifyEnum);

TEST_F(ScopifyEnumTest, TriggersOnUnscopedEnumDecl) {
  FileName = "Test.hpp";
  // Not available for scoped enum.
  EXPECT_UNAVAILABLE(R"cpp(enum class ^E { V };)cpp");

  // Not available for non-definition.
  EXPECT_UNAVAILABLE(R"cpp(
enum E { V };
enum ^E;
)cpp");
}

TEST_F(ScopifyEnumTest, ApplyTest) {
  std::string Original = R"cpp(
enum ^E { EV1, EV2, EV3 };
enum E;
E func(E in)
{
  E out = EV1;
  if (in == EV2)
    out = E::EV3;
  return out;
}
)cpp";
  std::string Expected = R"cpp(
enum class E { EV1, EV2, EV3 };
enum class E;
E func(E in)
{
  E out = E::EV1;
  if (in == E::EV2)
    out = E::EV3;
  return out;
}
)cpp";
  FileName = "Test.cpp";
  SCOPED_TRACE(Original);
  EXPECT_EQ(apply(Original), Expected);
}

} // namespace
} // namespace clang::clangd
