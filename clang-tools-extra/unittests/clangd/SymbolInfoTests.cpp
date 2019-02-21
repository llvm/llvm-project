//===-- SymbolInfoTests.cpp  -----------------------*- C++ -*--------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Annotations.h"
#include "ClangdUnit.h"
#include "Compiler.h"
#include "Matchers.h"
#include "SyncAPI.h"
#include "TestFS.h"
#include "TestTU.h"
#include "XRefs.h"
#include "index/FileIndex.h"
#include "index/SymbolCollector.h"
#include "clang/Index/IndexingAction.h"
#include "llvm/Support/Path.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

using testing::ElementsAreArray;

auto CreateExpectedSymbolDetails = [](const std::string &name,
                                      const std::string &container,
                                      const std::string &USR) {
  return SymbolDetails{name, container, USR, SymbolID(USR)};
};

TEST(SymbolInfoTests, All) {
  std::pair<const char *, std::vector<SymbolDetails>>
      TestInputExpectedOutput[] = {
          {
              R"cpp( // Simple function reference - declaration
          void foo();
          int bar() {
            fo^o();
          }
        )cpp",
              {CreateExpectedSymbolDetails("foo", "", "c:@F@foo#")}},
          {
              R"cpp( // Simple function reference - definition
          void foo() {}
          int bar() {
            fo^o();
          }
        )cpp",
              {CreateExpectedSymbolDetails("foo", "", "c:@F@foo#")}},
          {
              R"cpp( // Function in namespace reference
          namespace bar {
            void foo();
            int baz() {
              fo^o();
            }
          }
        )cpp",
              {CreateExpectedSymbolDetails("foo", "bar::", "c:@N@bar@F@foo#")}},
          {
              R"cpp( // Function in different namespace reference
          namespace bar {
            void foo();
          }
          namespace barbar {
            int baz() {
              bar::fo^o();
            }
          }
        )cpp",
              {CreateExpectedSymbolDetails("foo", "bar::", "c:@N@bar@F@foo#")}},
          {
              R"cpp( // Function in global namespace reference
          void foo();
          namespace Nbar {
            namespace Nbaz {
              int baz() {
                ::fo^o();
              }
            }
          }
        )cpp",
              {CreateExpectedSymbolDetails("foo", "", "c:@F@foo#")}},
          {
              R"cpp( // Function in anonymous namespace reference
          namespace {
            void foo();
          }
          namespace barbar {
            int baz() {
              fo^o();
            }
          }
        )cpp",
              {CreateExpectedSymbolDetails("foo", "(anonymous)",
                                           "c:TestTU.cpp@aN@F@foo#")}},
          {
              R"cpp( // Function reference - ADL
          namespace bar {
            struct BarType {};
            void foo(const BarType&);
          }
          namespace barbar {
            int baz() {
              bar::BarType b;
              fo^o(b);
            }
          }
        )cpp",
              {CreateExpectedSymbolDetails(
                  "foo", "bar::", "c:@N@bar@F@foo#&1$@N@bar@S@BarType#")}},
          {
              R"cpp( // Global value reference
          int value;
          void foo(int) { }
          void bar() {
            foo(val^ue);
          }
        )cpp",
              {CreateExpectedSymbolDetails("value", "", "c:@value")}},
          {
              R"cpp( // Local value reference
          void foo() { int aaa; int bbb = aa^a; }
        )cpp",
              {CreateExpectedSymbolDetails("aaa", "foo",
                                           "c:TestTU.cpp@49@F@foo#@aaa")}},
          {
              R"cpp( // Function param
          void bar(int aaa) {
            int bbb = a^aa;
          }
        )cpp",
              {CreateExpectedSymbolDetails("aaa", "bar",
                                           "c:TestTU.cpp@38@F@bar#I#@aaa")}},
          {
              R"cpp( // Lambda capture
          int ii;
          auto lam = [ii]() {
            return i^i;
          };
        )cpp",
              {CreateExpectedSymbolDetails("ii", "", "c:@ii")}},
          {
              R"cpp( // Macro reference
          #define MACRO 5\nint i = MAC^RO;
        )cpp",
              {CreateExpectedSymbolDetails("MACRO", "",
                                           "c:TestTU.cpp@38@macro@MACRO")}},
          {
              R"cpp( // Macro reference
          #define MACRO 5\nint i = MACRO^;
        )cpp",
              {CreateExpectedSymbolDetails("MACRO", "",
                                           "c:TestTU.cpp@38@macro@MACRO")}},
          {
              R"cpp( // Multiple symbols returned - using overloaded function name
          void foo() {}
          void foo(bool) {}
          void foo(int) {}
          namespace bar {
            using ::fo^o;
          }
        )cpp",
              {CreateExpectedSymbolDetails("foo", "", "c:@F@foo#"),
               CreateExpectedSymbolDetails("foo", "", "c:@F@foo#b#"),
               CreateExpectedSymbolDetails("foo", "", "c:@F@foo#I#")}},
          {
              R"cpp( // Multiple symbols returned - implicit conversion
          struct foo {};
          struct bar {
            bar(const foo&) {}
          };
          void func_baz1(bar) {}
          void func_baz2() {
            foo ff;
            func_baz1(f^f);
          }
        )cpp",
              {
                  CreateExpectedSymbolDetails(
                      "ff", "func_baz2", "c:TestTU.cpp@218@F@func_baz2#@ff"),
                  CreateExpectedSymbolDetails(
                      "bar", "bar::", "c:@S@bar@F@bar#&1$@S@foo#"),
              }},
          {
              R"cpp( // Type reference - declaration
          struct foo;
          void bar(fo^o*);
        )cpp",
              {CreateExpectedSymbolDetails("foo", "", "c:@S@foo")}},
          {
              R"cpp( // Type reference - definition
          struct foo {};
          void bar(fo^o*);
        )cpp",
              {CreateExpectedSymbolDetails("foo", "", "c:@S@foo")}},
          {
              R"cpp( // Type Reference - template argumen
          struct foo {};
          template<class T> struct bar {};
          void baz() {
            bar<fo^o> b;
          }
        )cpp",
              {CreateExpectedSymbolDetails("foo", "", "c:@S@foo")}},
          {
              R"cpp( // Template parameter reference - type param
          template<class TT> struct bar {
            T^T t;
          };
        )cpp",
              {CreateExpectedSymbolDetails("TT", "bar::", "c:TestTU.cpp@65")}},
          {
              R"cpp( // Template parameter reference - type param
          template<int NN> struct bar {
            int a = N^N;
          };
        )cpp",
              {CreateExpectedSymbolDetails("NN", "bar::", "c:TestTU.cpp@65")}},
          {
              R"cpp( // Class member reference - objec
          struct foo {
            int aa;
          };
          void bar() {
            foo f;
            f.a^a;
          }
        )cpp",
              {CreateExpectedSymbolDetails("aa", "foo::", "c:@S@foo@FI@aa")}},
          {
              R"cpp( // Class member reference - pointer
          struct foo {
            int aa;
          };
          void bar() {
            &foo::a^a;
          }
        )cpp",
              {CreateExpectedSymbolDetails("aa", "foo::", "c:@S@foo@FI@aa")}},
          {
              R"cpp( // Class method reference - objec
          struct foo {
            void aa() {}
          };
          void bar() {
            foo f;
            f.a^a();
          }
        )cpp",
              {CreateExpectedSymbolDetails("aa", "foo::", "c:@S@foo@F@aa#")}},
          {
              R"cpp( // Class method reference - pointer
          struct foo {
            void aa() {}
          };
          void bar() {
            &foo::a^a;
          }
        )cpp",
              {CreateExpectedSymbolDetails("aa", "foo::", "c:@S@foo@F@aa#")}},
          {
              R"cpp( // Typedef
          typedef int foo;
          void bar() {
            fo^o a;
          }
        )cpp",
              {CreateExpectedSymbolDetails("foo", "", "c:TestTU.cpp@T@foo")}},
          {
              R"cpp( // Type alias
          using foo = int;
          void bar() {
            fo^o a;
          }
        )cpp",
              {CreateExpectedSymbolDetails("foo", "", "c:@foo")}},
          {
              R"cpp( // Namespace reference
          namespace foo {}
          using namespace fo^o;
        )cpp",
              {CreateExpectedSymbolDetails("foo", "", "c:@N@foo")}},
          {
              R"cpp( // Enum value reference
          enum foo { bar, baz };
          void f() {
            foo fff = ba^r;
          }
        )cpp",
              {CreateExpectedSymbolDetails("bar", "foo", "c:@E@foo@bar")}},
          {
              R"cpp( // Enum class value reference
          enum class foo { bar, baz };
          void f() {
            foo fff = foo::ba^r;
          }
        )cpp",
              {CreateExpectedSymbolDetails("bar", "foo::", "c:@E@foo@bar")}},
          {
              R"cpp( // Parameters in declarations
          void foo(int ba^r);
        )cpp",
              {CreateExpectedSymbolDetails("bar", "foo",
                                           "c:TestTU.cpp@50@F@foo#I#@bar")}},
          {
              R"cpp( // Type inferrence with auto keyword
          struct foo {};
          foo getfoo() { return foo{}; }
          void f() {
            au^to a = getfoo();
          }
        )cpp",
              {/* not implemented */}},
          {
              R"cpp( // decltype
          struct foo {};
          void f() {
            foo f;
            declt^ype(f);
          }
        )cpp",
              {/* not implemented */}},
      };

  for (const auto &T : TestInputExpectedOutput) {
    Annotations TestInput(T.first);
    auto AST = TestTU::withCode(TestInput.code()).build();

    EXPECT_THAT(getSymbolInfo(AST, TestInput.point()),
                ElementsAreArray(T.second))
        << T.first;
  }
}

} // namespace
} // namespace clangd
} // namespace clang
