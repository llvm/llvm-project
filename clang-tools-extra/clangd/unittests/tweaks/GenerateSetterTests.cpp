//===-- GenerateSetterTests.cpp -------------------------------------------===//
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

TWEAK_TEST(GenerateSetter);

TEST_F(GenerateSetterTest, Availability) {
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
  EXPECT_UNAVAILABLE("struct S { int setX(); int ^x, y; };");
  // unavailable on constant type:
  EXPECT_UNAVAILABLE("class S { const int ^x, y; };");
  // unavailable on static member:
  EXPECT_UNAVAILABLE("struct S { static int ^x; };");
}

TEST_F(GenerateSetterTest, Edits) {
  auto RunSetterTest = [&](llvm::StringRef SetterPrefix,
                           llvm::StringMap<std::string> Options,
                           llvm::StringRef Input, llvm::StringRef Expected) {
    Config Cfg;
    if (!SetterPrefix.empty())
      Cfg.Style.SetterPrefix = SetterPrefix.str();

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
    struct Bigfoo {
      long a;
      long b;
      long c;
      long d;
    };
  )cpp";

  // Comply with style configuration:
  RunSetterTest(
      "put",
      {
          {"PublicMemberPrefix", "m_"},
          {"PublicMemberSuffix", "_s"},
          {"PublicMethodCase", "CamelCase"},
      },
      "struct S{ int m_m^ember_s;};",
      "struct S{ void PutMember(int member) { m_member_s = member; }\n"
      "int m_member_s;};");

  RunSetterTest("set",
                {
                    {"PrivateMemberPrefix", "_"},
                    {"PrivateMemberSuffix", ""},
                    {"PublicMethodCase", "camelBack"},
                },
                "class S { int ^_member; };",
                "class S { int _member; public:\n"
                "void setMember(int member) { _member = member; }\n};");

  // Use const-ref on non trivially copiable member:
  RunSetterTest(
      "",
      {
          {"PrivateMemberPrefix", "m_"},
          {"PrivateMemberSuffix", ""},
          {"PublicMethodCase", "camelBack"},
      },
      "class S { Bigfoo ^m_bigfoo; };",
      "class S { Bigfoo m_bigfoo; public:\n"
      "void setBigfoo(const Bigfoo &bigfoo) { m_bigfoo = bigfoo; }\n};");

  // Use const-ref on non trivially copiable reference member (don't duplicate
  // the reference):
  RunSetterTest(
      "",
      {
          {"PrivateMemberPrefix", "m_"},
          {"PrivateMemberSuffix", ""},
          {"PublicMethodCase", "camelBack"},
      },
      "class S { Bigfoo &^m_bigfoo; };",
      "class S { Bigfoo &m_bigfoo; public:\n"
      "void setBigfoo(const Bigfoo &bigfoo) { m_bigfoo = bigfoo; }\n};");

  // Member prefix and suffix comply with style precedence:
  RunSetterTest("",
                {
                    {"PrivateMemberPrefix", "pre_"},
                    {"PrivateMemberSuffix", "_post"},
                    {"MemberPrefix", "not_used"},
                    {"MemberSuffix", "not_used"},
                    {"MethodCase", "lower_case"},
                },
                "class S { public: S(); private: Foo *pre_foo^_post; };",
                "class S { public: S(); void set_foo(Foo * foo) { "
                "pre_foo_post = foo; }\n"
                "private: Foo *pre_foo_post; };");

  // Method case comply with style precedence:
  RunSetterTest(
      "set",
      {
          {"ProtectedMemberPrefix", "pre_"},
          {"ProtectedMemberSuffix", "_post"},
          {"PublicMethodCase", "Leading_upper_snake_case"},
          {"MethodCase", "lower_case"},
      },
      "class S { public: void Set_foo(const Foo &foo); protected: Foo "
      "&pre_foo_^post; };",
      "unavailable");

  // Don't comply to unrelated member suffix, then fallback to default setter
  // prefix
  RunSetterTest("",
                {
                    {"ProtectedMemberSuffix", "_post"},
                },
                "class S { public: S(); private: Foo foo^_post; };",
                "class S { public: S(); void setFoo_post(Foo newfoo_post) { "
                "foo_post = newfoo_post; }\n"
                "private: Foo foo_post; };");
}

} // namespace
} // namespace clangd
} // namespace clang
