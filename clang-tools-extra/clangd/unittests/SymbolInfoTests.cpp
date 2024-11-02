//===-- SymbolInfoTests.cpp  -----------------------*- C++ -*--------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Annotations.h"
#include "ParsedAST.h"
#include "TestTU.h"
#include "XRefs.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

using ::testing::UnorderedElementsAreArray;

// Partial SymbolDetails with the rest filled in at testing time.
struct ExpectedSymbolDetails {
  std::string Name;
  std::string Container;
  std::string USR;
  const char *DeclMarker = nullptr;
  const char *DefMarker = nullptr;
};

TEST(SymbolInfoTests, All) {
  std::pair<const char *, std::vector<ExpectedSymbolDetails>>
      TestInputExpectedOutput[] = {
          {
              R"cpp( // Simple function reference - declaration
          void $decl[[foo]]();
          int bar() {
            fo^o();
          }
        )cpp",
              {ExpectedSymbolDetails{"foo", "", "c:@F@foo#", "decl"}}},
          {
              R"cpp( // Simple function reference - definition
          void $def[[foo]]() {}
          int bar() {
            fo^o();
          }
        )cpp",
              {ExpectedSymbolDetails{"foo", "", "c:@F@foo#", "def", "def"}}},
          {
              R"cpp( // Simple function reference - decl and def
          void $decl[[foo]]();
          void $def[[foo]]() {}
          int bar() {
            fo^o();
          }
        )cpp",
              {ExpectedSymbolDetails{"foo", "", "c:@F@foo#", "decl", "def"}}},
          {
              R"cpp( // Simple class reference - decl and def
          @interface $decl[[Foo]]
          @end
          @implementation $def[[Foo]]
          @end
          void doSomething(F^oo *obj) {}
        )cpp",
              {ExpectedSymbolDetails{"Foo", "", "c:objc(cs)Foo", "decl",
                                     "def"}}},
          {
              R"cpp( // Simple method reference - decl and def
          @interface Foo
          - (void)$decl[[foo]];
          @end
          @implementation Foo
          - (void)$def[[fo^o]] {}
          @end
        )cpp",
              {ExpectedSymbolDetails{"foo", "Foo::", "c:objc(cs)Foo(im)foo",
                                     "decl", "def"}}},
          {
              R"cpp( // Function in namespace reference
          namespace bar {
            void $decl[[foo]]();
            int baz() {
              fo^o();
            }
          }
        )cpp",
              {ExpectedSymbolDetails{"foo", "bar::", "c:@N@bar@F@foo#",
                                     "decl"}}},
          {
              R"cpp( // Function in different namespace reference
          namespace bar {
            void $decl[[foo]]();
          }
          namespace barbar {
            int baz() {
              bar::fo^o();
            }
          }
        )cpp",
              {ExpectedSymbolDetails{"foo", "bar::", "c:@N@bar@F@foo#",
                                     "decl"}}},
          {
              R"cpp( // Function in global namespace reference
          void $decl[[foo]]();
          namespace Nbar {
            namespace Nbaz {
              int baz() {
                ::fo^o();
              }
            }
          }
        )cpp",
              {ExpectedSymbolDetails{"foo", "", "c:@F@foo#", "decl"}}},
          {
              R"cpp( // Function in anonymous namespace reference
          namespace {
            void $decl[[foo]]();
          }
          namespace barbar {
            int baz() {
              fo^o();
            }
          }
        )cpp",
              {ExpectedSymbolDetails{"foo", "(anonymous)",
                                     "c:TestTU.cpp@aN@F@foo#", "decl"}}},
          {
              R"cpp( // Function reference - ADL
          namespace bar {
            struct BarType {};
            void $decl[[foo]](const BarType&);
          }
          namespace barbar {
            int baz() {
              bar::BarType b;
              fo^o(b);
            }
          }
        )cpp",
              {ExpectedSymbolDetails{
                  "foo", "bar::", "c:@N@bar@F@foo#&1$@N@bar@S@BarType#",
                  "decl"}}},
          {
              R"cpp( // Global value reference
          int $def[[value]];
          void foo(int) { }
          void bar() {
            foo(val^ue);
          }
        )cpp",
              {ExpectedSymbolDetails{"value", "", "c:@value", "def", "def"}}},
          {
              R"cpp( // Local value reference
          void foo() { int $def[[aaa]]; int bbb = aa^a; }
        )cpp",
              {ExpectedSymbolDetails{"aaa", "foo", "c:TestTU.cpp@49@F@foo#@aaa",
                                     "def", "def"}}},
          {
              R"cpp( // Function param
          void bar(int $def[[aaa]]) {
            int bbb = a^aa;
          }
        )cpp",
              {ExpectedSymbolDetails{
                  "aaa", "bar", "c:TestTU.cpp@38@F@bar#I#@aaa", "def", "def"}}},
          {
              R"cpp( // Lambda capture
          void foo() {
            int $def[[ii]];
            auto lam = [ii]() {
              return i^i;
            };
          }
        )cpp",
              {ExpectedSymbolDetails{"ii", "foo", "c:TestTU.cpp@54@F@foo#@ii",
                                     "def", "def"}}},
          {
              R"cpp( // Macro reference
          #define MACRO 5\nint i = MAC^RO;
        )cpp",
              {ExpectedSymbolDetails{"MACRO", "",
                                     "c:TestTU.cpp@38@macro@MACRO"}}},
          {
              R"cpp( // Macro reference
          #define MACRO 5\nint i = MACRO^;
        )cpp",
              {ExpectedSymbolDetails{"MACRO", "",
                                     "c:TestTU.cpp@38@macro@MACRO"}}},
          {
              R"cpp( // Multiple symbols returned - using overloaded function name
          void $def[[foo]]() {}
          void $def_bool[[foo]](bool) {}
          void $def_int[[foo]](int) {}
          namespace bar {
            using ::$decl[[fo^o]];
          }
        )cpp",
              {ExpectedSymbolDetails{"foo", "", "c:@F@foo#", "def", "def"},
               ExpectedSymbolDetails{"foo", "", "c:@F@foo#b#", "def_bool",
                                     "def_bool"},
               ExpectedSymbolDetails{"foo", "", "c:@F@foo#I#", "def_int",
                                     "def_int"},
               ExpectedSymbolDetails{"foo", "bar::", "c:@N@bar@UD@foo",
                                     "decl"}}},
          {
              R"cpp( // Multiple symbols returned - implicit conversion
          struct foo {};
          struct bar {
            bar(const foo&) {}
          };
          void func_baz1(bar) {}
          void func_baz2() {
            foo $def[[ff]];
            func_baz1(f^f);
          }
        )cpp",
              {ExpectedSymbolDetails{"ff", "func_baz2",
                                     "c:TestTU.cpp@218@F@func_baz2#@ff", "def",
                                     "def"}}},
          {
              R"cpp( // Type reference - declaration
          struct $decl[[foo]];
          void bar(fo^o*);
        )cpp",
              {ExpectedSymbolDetails{"foo", "", "c:@S@foo", "decl"}}},
          {
              R"cpp( // Type reference - definition
          struct $def[[foo]] {};
          void bar(fo^o*);
        )cpp",
              {ExpectedSymbolDetails{"foo", "", "c:@S@foo", "def", "def"}}},
          {
              R"cpp( // Type Reference - template argument
          struct $def[[foo]] {};
          template<class T> struct bar {};
          void baz() {
            bar<fo^o> b;
          }
        )cpp",
              {ExpectedSymbolDetails{"foo", "", "c:@S@foo", "def", "def"}}},
          {
              R"cpp( // Template parameter reference - type param
          template<class $def[[TT]]> struct bar {
            T^T t;
          };
        )cpp",
              {ExpectedSymbolDetails{"TT", "bar::", "c:TestTU.cpp@65", "def",
                                     "def"}}},
          {
              R"cpp( // Template parameter reference - type param
          template<int $def[[NN]]> struct bar {
            int a = N^N;
          };
        )cpp",
              {ExpectedSymbolDetails{"NN", "bar::", "c:TestTU.cpp@65", "def",
                                     "def"}}},
          {
              R"cpp( // Class member reference - objec
          struct foo {
            int $def[[aa]];
          };
          void bar() {
            foo f;
            f.a^a;
          }
        )cpp",
              {ExpectedSymbolDetails{"aa", "foo::", "c:@S@foo@FI@aa", "def",
                                     "def"}}},
          {
              R"cpp( // Class member reference - pointer
          struct foo {
            int $def[[aa]];
          };
          void bar() {
            &foo::a^a;
          }
        )cpp",
              {ExpectedSymbolDetails{"aa", "foo::", "c:@S@foo@FI@aa", "def",
                                     "def"}}},
          {
              R"cpp( // Class method reference - objec
          struct foo {
            void $def[[aa]]() {}
          };
          void bar() {
            foo f;
            f.a^a();
          }
        )cpp",
              {ExpectedSymbolDetails{"aa", "foo::", "c:@S@foo@F@aa#", "def",
                                     "def"}}},
          {
              R"cpp( // Class method reference - pointer
          struct foo {
            void $def[[aa]]() {}
          };
          void bar() {
            &foo::a^a;
          }
        )cpp",
              {ExpectedSymbolDetails{"aa", "foo::", "c:@S@foo@F@aa#", "def",
                                     "def"}}},
          {
              R"cpp( // Typedef
          typedef int $decl[[foo]];
          void bar() {
            fo^o a;
          }
        )cpp",
              {ExpectedSymbolDetails{"foo", "", "c:TestTU.cpp@T@foo", "decl"}}},
          {
              R"cpp( // Type alias
          using $decl[[foo]] = int;
          void bar() {
            fo^o a;
          }
        )cpp",
              {ExpectedSymbolDetails{"foo", "", "c:@foo", "decl"}}},
          {
              R"cpp( // Namespace reference
          namespace $decl[[foo]] {}
          using namespace fo^o;
        )cpp",
              {ExpectedSymbolDetails{"foo", "", "c:@N@foo", "decl"}}},
          {
              R"cpp( // Enum value reference
          enum foo { $def[[bar]], baz };
          void f() {
            foo fff = ba^r;
          }
        )cpp",
              {ExpectedSymbolDetails{"bar", "foo", "c:@E@foo@bar", "def",
                                     "def"}}},
          {
              R"cpp( // Enum class value reference
          enum class foo { $def[[bar]], baz };
          void f() {
            foo fff = foo::ba^r;
          }
        )cpp",
              {ExpectedSymbolDetails{"bar", "foo::", "c:@E@foo@bar", "def",
                                     "def"}}},
          {
              R"cpp( // Parameters in declarations
          void foo(int $def[[ba^r]]);
        )cpp",
              {ExpectedSymbolDetails{
                  "bar", "foo", "c:TestTU.cpp@50@F@foo#I#@bar", "def", "def"}}},
          {
              R"cpp( // Type inference with auto keyword
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
    TestTU TU;
    TU.Code = std::string(TestInput.code());
    TU.ExtraArgs.push_back("-xobjective-c++");
    auto AST = TU.build();

    std::vector<SymbolDetails> Expected;
    for (const auto &Sym : T.second) {
      llvm::Optional<Location> Decl, Def;
      if (Sym.DeclMarker)
        Decl = Location{URIForFile::canonicalize(testPath(TU.Filename), ""),
                        TestInput.range(Sym.DeclMarker)};
      if (Sym.DefMarker)
        Def = Location{URIForFile::canonicalize(testPath(TU.Filename), ""),
                       TestInput.range(Sym.DefMarker)};
      Expected.push_back(
          {Sym.Name, Sym.Container, Sym.USR, SymbolID(Sym.USR), Decl, Def});
    }

    EXPECT_THAT(getSymbolInfo(AST, TestInput.point()),
                UnorderedElementsAreArray(Expected))
        << T.first;
  }
}

} // namespace
} // namespace clangd
} // namespace clang
