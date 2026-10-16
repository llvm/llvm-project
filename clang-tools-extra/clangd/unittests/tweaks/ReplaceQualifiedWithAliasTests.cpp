//===-- ReplaceQualifiedWithAliasTests.cpp -----------------------*- C++
//-*-===//
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

TWEAK_TEST(ReplaceQualifiedWithAlias);

TEST_F(ReplaceQualifiedWithAliasTest, Prepare) {
  const char *Header = R"cpp(
namespace ns { struct Foo {}; }
)cpp";

  EXPECT_AVAILABLE(std::string(Header) + R"cpp(
using Y = ns::F^oo;
ns::Foo A;
)cpp");

  EXPECT_AVAILABLE(std::string(Header) + R"cpp(
using ns::F^oo;
ns::Foo A;
)cpp");

  EXPECT_UNAVAILABLE(std::string(Header) + R"cpp(
using Y = ns::Foo;^
ns::Foo A;
)cpp");
}

TEST_F(ReplaceQualifiedWithAliasTest, Apply) {
  EXPECT_EQ(apply(R"cpp(
namespace ns {
struct Foo {};
}

using Y = ns::F^oo;

ns::Foo A;
void f() {
  ns::Foo B;
}
)cpp"),
            R"cpp(
namespace ns {
struct Foo {};
}

using Y = ns::Foo;

Y A;
void f() {
  Y B;
}
)cpp");
}

TEST_F(ReplaceQualifiedWithAliasTest, ApplyInsideMacroArgument) {
  EXPECT_EQ(apply(R"cpp(
namespace ns {
struct Foo {};
}

#define ID(X) X
using Y = ns::F^oo;

ID(ns::Foo) A;
)cpp"),
            R"cpp(
namespace ns {
struct Foo {};
}

#define ID(X) X
using Y = ns::Foo;

ID(Y) A;
)cpp");
}

TEST_F(ReplaceQualifiedWithAliasTest, KeepAliasDeclarationUnchanged) {
  EXPECT_EQ(apply(R"cpp(
namespace ns {
template <typename T> struct Foo {};
}

using Y = ns::F^oo<int>;

ns::Foo<int> A;
)cpp"),
            R"cpp(
namespace ns {
template <typename T> struct Foo {};
}

using Y = ns::Foo<int>;

Y A;
)cpp");
}

TEST_F(ReplaceQualifiedWithAliasTest, ApplyNestedNamespaces) {
  EXPECT_EQ(apply(R"cpp(
namespace a {
namespace b {
struct Foo {};
}
}

using Y = a::b::F^oo;

a::b::Foo A;
::a::b::Foo B;
)cpp"),
            R"cpp(
namespace a {
namespace b {
struct Foo {};
}
}

using Y = a::b::Foo;

Y A;
Y B;
)cpp");
}

TEST_F(ReplaceQualifiedWithAliasTest, ApplyWithCompetingAlias) {
  EXPECT_EQ(apply(R"cpp(
namespace ns {
struct Foo {};
}

using Y = ns::F^oo;
using Z = ns::Foo;

ns::Foo A;
)cpp"),
            R"cpp(
namespace ns {
struct Foo {};
}

using Y = ns::Foo;
using Z = Y;

Y A;
)cpp");
}

TEST_F(ReplaceQualifiedWithAliasTest, MustNotReplaceExistingUsingDecl) {
  EXPECT_EQ(apply(R"cpp(
namespace b {
    int test1;
}

namespace d {
    int test1;
}

namespace a {
    using b::test1;

    void foo() {
        b::test1 = 1;
        b::test1 = 1;
    }
} // namespace a

namespace c {
    using d::test1;
} // namespace c

using namespace a;
using namespace b;

void foo() {
    b::te^st1 = 1;
    a::test1 = 1;
    c::test1 = 1;
}

)cpp"),
            R"cpp(
namespace b {
    int test1;
}

namespace d {
    int test1;
}

namespace a {
    using b::test1;

    void foo() {
        test1 = 1;
        test1 = 1;
    }
} // namespace a

namespace c {
    using d::test1;
} // namespace c

using namespace a;
using namespace b;

void foo() {
    test1 = 1;
    a::test1 = 1;
    c::test1 = 1;
}

)cpp");
}

} // namespace
} // namespace clangd
} // namespace clang
