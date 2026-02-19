//===-- GenerateGetterTests.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Config.h"
#include "TweakTesting.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

TWEAK_TEST(GenerateGetter);

TEST_F(GenerateGetterTest, Availability) {
  // available on class member:
  EXPECT_AVAILABLE("struct S { int ^x, y; };");
  EXPECT_AVAILABLE("class S { private: int ^x; };");
  EXPECT_AVAILABLE("class S { protected: int ^x; };");
  EXPECT_AVAILABLE("union S { int ^x; };");
  EXPECT_AVAILABLE("struct S { int ^x = 0; };");
  // available on forward member:
  EXPECT_AVAILABLE("/*error-ok*/class Forward; class A { Forward ^f; };");
  // available on pointer type:
  EXPECT_AVAILABLE("class Forward; class A { Forward *^f; };");
  // available on reference type:
  EXPECT_AVAILABLE("class A { int &^f; };");

  // unavailable outside class member:
  EXPECT_UNAVAILABLE("^struct ^S ^{ int f^oo(); ^int x, y; };");
  // unavailable if method already exists:
  EXPECT_UNAVAILABLE("struct S { int getX(); int ^x, y; };");
  // unavailable on constant type:
  EXPECT_UNAVAILABLE("class S { const int ^x, y; };");
  // unavailable on static member:
  EXPECT_UNAVAILABLE("struct S { static int ^x; };");
}

TEST_F(GenerateGetterTest, Edits) {
  auto RunGetterTest = [&](llvm::StringRef GetterPrefix,
                           llvm::StringMap<std::string> Options,
                           llvm::StringRef Input, llvm::StringRef Expected) {
    Config Cfg;
    if (!GetterPrefix.empty())
      Cfg.Style.GetterPrefix = GetterPrefix.str();

    for (auto &KV : Options)
      Cfg.Diagnostics.ClangTidy.CheckOptions.insert_or_assign(
          ("readability-identifier-naming." + KV.getKey()).str(),
          KV.getValue());

    WithContextValue WithCfg(Config::Key, std::move(Cfg));
    EXPECT_EQ(apply(Input.str()), Expected.str());
  };

  Header = R"cpp(
    struct Foo {
      char a;
      char b;
      char c;
    };
  )cpp";

  // Comply with style configuration:
  RunGetterTest("summon",
                {
                    {"PublicMemberPrefix", "m_"},
                    {"PublicMemberSuffix", "_s"},
                    {"PublicMethodCase", "CamelCase"},
                },
                "struct S{ int m_m^ember_s;};",
                "struct S{ int SummonMember() const { return m_member_s; }\n"
                "int m_member_s;};");

  RunGetterTest("get",
                {
                    {"PrivateMemberPrefix", "_"},
                    {"PrivateMemberSuffix", ""},
                    {"PublicMethodCase", "camelBack"},
                },
                "class S { int ^_member; };",
                "class S { int _member; public:\n"
                "int getMember() const { return _member; }\n};");

  // Member prefix and suffix comply with style precedence:
  RunGetterTest(
      "",
      {
          {"PrivateMemberPrefix", "pre_"},
          {"PrivateMemberSuffix", "_post"},
          {"MemberPrefix", "not_used"},
          {"MemberSuffix", "not_used"},
          {"MethodCase", "lower_case"},
      },
      "class S { public: S(); private: Foo *pre_foo^_post; };",
      "class S { public: S(); Foo * foo() const { return pre_foo_post; }\n"
      "private: Foo *pre_foo_post; };");

  // Method case comply with style precedence:
  RunGetterTest("get",
                {
                    {"ProtectedMemberPrefix", "pre_"},
                    {"ProtectedMemberSuffix", "_post"},
                    {"PublicMethodCase", "Leading_upper_snake_case"},
                    {"MethodCase", "lower_case"},
                },
                "class S { public: Foo &Get_foo(); protected: Foo "
                "&pre_foo_^post; };",
                "unavailable");

  // Don't comply to unrelated member suffix, then fallback to default getter
  // prefix
  RunGetterTest(
      "",
      {
          {"ProtectedMemberSuffix", "_post"},
      },
      "class S { public: S(); private: Foo foo^_post; };",
      "class S { public: S(); Foo getFoo_post() const { return foo_post; }\n"
      "private: Foo foo_post; };");
}

} // namespace
} // namespace clangd
} // namespace clang
