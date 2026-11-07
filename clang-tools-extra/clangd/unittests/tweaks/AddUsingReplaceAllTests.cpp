//===-- AddUsingReplaceAllTests.cpp -----------------------------*- C++ -*-===//
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

TWEAK_TEST(AddUsingReplaceAll);

TEST_F(AddUsingReplaceAllTest, Prepare) {
  const char *Header = R"cpp(namespace ns { struct SomeStruct {}; })cpp";

  EXPECT_AVAILABLE(std::string(Header) + R"cpp(
ns::SomeStruct bar() {
  n^s::SomeStruct x;
  return x;
})cpp");
  EXPECT_UNAVAILABLE(std::string(Header) + R"cpp(
ns::SomeStruct bar() {
  ns::SomeStruct ^x;
  return x;
})cpp");

  EXPECT_UNAVAILABLE(R"cpp(
struct S {
  static void f();
};
void fun() {
  S::f^();
}
)cpp");
}

TEST_F(AddUsingReplaceAllTest, Apply) {
  ExtraFiles["test.hpp"] = R"cpp(
namespace one {
namespace two {
void ff();
}
})cpp";

  EXPECT_EQ(apply(R"cpp(
#include "test.hpp"

void fun() {
  one::two::f^f();
  one::two::ff();
}

void other() {
  one::two::ff();
}
)cpp"),
            R"cpp(
#include "test.hpp"

using one::two::ff;

void fun() {
  ff();
  ff();
}

void other() {
  ff();
}
)cpp");
}

TEST_F(AddUsingReplaceAllTest, ApplyInsideMacroArgument) {
  ExtraFiles["test.hpp"] = R"cpp(
namespace one {
namespace two {
void ff();
}
})cpp";

  EXPECT_EQ(apply(R"cpp(
#include "test.hpp"
#define CALL(name) name()

void fun() {
  CALL(one::two::f^f);
  one::two::ff();
}
)cpp"),
            R"cpp(
#include "test.hpp"
#define CALL(name) name()

using one::two::ff;

void fun() {
  CALL(ff);
  ff();
}
)cpp");
}

TEST_F(AddUsingReplaceAllTest, ApplyWithGlobalQualifierAndNestedNamespaceDecl) {
  ExtraFiles["test.hpp"] = R"cpp(
namespace one::two {
void ff();
}
)cpp";

  EXPECT_EQ(apply(R"cpp(
#include "test.hpp"

void fun() {
  ::one::two::f^f();
  one::two::ff();
}
)cpp"),
            R"cpp(
#include "test.hpp"

using ::one::two::ff;

void fun() {
  ff();
  ff();
}
)cpp");
}

TEST_F(AddUsingReplaceAllTest, ApplyWithMultipleMacros) {
  ExtraFiles["test.hpp"] = R"cpp(
namespace one {
namespace two {
void ff();
}
}
)cpp";

  EXPECT_EQ(apply(R"cpp(
#include "test.hpp"
#define ID(X) X
#define CALL(name) name()
#define CALL_FF one::two::ff()

void fun() {
  CALL(ID(one::two::f^f));
  CALL_FF;
  one::two::ff();
}
)cpp"),
            R"cpp(
#include "test.hpp"
#define ID(X) X
#define CALL(name) name()
#define CALL_FF one::two::ff()

using one::two::ff;

void fun() {
  CALL(ID(ff));
  CALL_FF;
  ff();
}
)cpp");
}

} // namespace
} // namespace clangd
} // namespace clang
