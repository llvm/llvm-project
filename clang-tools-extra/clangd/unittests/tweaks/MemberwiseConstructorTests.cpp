//===-- MemberwiseConstructorTests.cpp ------------------------------------===//
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
using testing::AllOf;
using testing::Eq;
using testing::HasSubstr;
using testing::Not;

TWEAK_TEST(MemberwiseConstructor);

TEST_F(MemberwiseConstructorTest, Availability) {
  EXPECT_AVAILABLE("^struct ^S ^{ int x, y; };");
  // Verify no crashes on incomplete member fields.
  EXPECT_UNAVAILABLE("/*error-ok*/class Forward; class ^A { Forward f;}");
  EXPECT_UNAVAILABLE("struct S { ^int ^x, y; }; struct ^S;");
  EXPECT_UNAVAILABLE("struct ^S {};");
  EXPECT_UNAVAILABLE("union ^S { int x; };");
  EXPECT_UNAVAILABLE("struct ^S { int x = 0; };");
  EXPECT_UNAVAILABLE("struct ^S { struct { int x; }; };");
  EXPECT_UNAVAILABLE("struct ^{ int x; } e;");
}

TEST_F(MemberwiseConstructorTest, Edits) {
  Header = R"cpp(
    struct Move {
      Move(Move&&) = default;
      Move(const Move&) = delete;
    };
    struct Copy {
      Copy(Copy&&) = delete;
      Copy(const Copy&);
    };
  )cpp";
  EXPECT_EQ(apply("struct ^S{Move M; Copy C; int I; int J=4;};"),
            "struct S{"
            "S(Move M, const Copy &C, int I) : M(std::move(M)), C(C), I(I) {}\n"
            "Move M; Copy C; int I; int J=4;};");
}

TEST_F(MemberwiseConstructorTest, FieldTreatment) {
  Header = R"cpp(
    struct MoveOnly {
      MoveOnly(MoveOnly&&) = default;
      MoveOnly(const MoveOnly&) = delete;
    };
    struct CopyOnly {
      CopyOnly(CopyOnly&&) = delete;
      CopyOnly(const CopyOnly&);
    };
    struct CopyTrivial {
      CopyTrivial(CopyTrivial&&) = default;
      CopyTrivial(const CopyTrivial&) = default;
    };
    struct Immovable {
      Immovable(Immovable&&) = delete;
      Immovable(const Immovable&) = delete;
    };
    template <typename T>
    struct Traits { using Type = typename T::Type; };
    using IntAlias = int;
  )cpp";

  auto Fail = Eq("unavailable");
  auto Move = HasSubstr(": Member(std::move(Member))");
  auto CopyRef = AllOf(HasSubstr("S(const "), HasSubstr(": Member(Member)"));
  auto Copy = AllOf(Not(HasSubstr("S(const ")), HasSubstr(": Member(Member)"));
  auto With = [](llvm::StringRef Type) {
    return ("struct ^S { " + Type + " Member; };").str();
  };

  EXPECT_THAT(apply(With("Immovable")), Fail);
  EXPECT_THAT(apply(With("MoveOnly")), Move);
  EXPECT_THAT(apply(With("CopyOnly")), CopyRef);
  EXPECT_THAT(apply(With("CopyTrivial")), Copy);
  EXPECT_THAT(apply(With("int")), Copy);
  EXPECT_THAT(apply(With("IntAlias")), Copy);
  EXPECT_THAT(apply(With("Immovable*")), Copy);
  EXPECT_THAT(apply(With("Immovable&")), Copy);

  EXPECT_THAT(apply("template <typename T>" + With("T")), Move);
  EXPECT_THAT(apply("template <typename T>" + With("typename Traits<T>::Type")),
              Move);
  EXPECT_THAT(apply("template <typename T>" + With("T*")), Copy);
}

} // namespace
} // namespace clangd
} // namespace clang
