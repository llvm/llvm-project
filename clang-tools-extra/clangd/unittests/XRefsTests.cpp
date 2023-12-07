//===-- XRefsTests.cpp  ---------------------------*- C++ -*--------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Annotations.h"
#include "AST.h"
#include "ParsedAST.h"
#include "Protocol.h"
#include "SourceCode.h"
#include "SyncAPI.h"
#include "TestFS.h"
#include "TestTU.h"
#include "XRefs.h"
#include "index/MemIndex.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ScopedPrinter.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <optional>
#include <string>
#include <vector>

namespace clang {
namespace clangd {
namespace {

using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::IsEmpty;
using ::testing::Matcher;
using ::testing::UnorderedElementsAre;
using ::testing::UnorderedElementsAreArray;
using ::testing::UnorderedPointwise;

std::string guard(llvm::StringRef Code) {
  return "#pragma once\n" + Code.str();
}

MATCHER_P2(FileRange, File, Range, "") {
  return Location{URIForFile::canonicalize(File, testRoot()), Range} == arg;
}
MATCHER(declRange, "") {
  const LocatedSymbol &Sym = ::testing::get<0>(arg);
  const Range &Range = ::testing::get<1>(arg);
  return Sym.PreferredDeclaration.range == Range;
}

// Extracts ranges from an annotated example, and constructs a matcher for a
// highlight set. Ranges should be named $read/$write as appropriate.
Matcher<const std::vector<DocumentHighlight> &>
highlightsFrom(const Annotations &Test) {
  std::vector<DocumentHighlight> Expected;
  auto Add = [&](const Range &R, DocumentHighlightKind K) {
    Expected.emplace_back();
    Expected.back().range = R;
    Expected.back().kind = K;
  };
  for (const auto &Range : Test.ranges())
    Add(Range, DocumentHighlightKind::Text);
  for (const auto &Range : Test.ranges("read"))
    Add(Range, DocumentHighlightKind::Read);
  for (const auto &Range : Test.ranges("write"))
    Add(Range, DocumentHighlightKind::Write);
  return UnorderedElementsAreArray(Expected);
}

TEST(HighlightsTest, All) {
  const char *Tests[] = {
      R"cpp(// Local variable
        int main() {
          int [[bonjour]];
          $write[[^bonjour]] = 2;
          int test1 = $read[[bonjour]];
        }
      )cpp",

      R"cpp(// Struct
        namespace ns1 {
        struct [[MyClass]] {
          static void foo([[MyClass]]*) {}
        };
        } // namespace ns1
        int main() {
          ns1::[[My^Class]]* Params;
        }
      )cpp",

      R"cpp(// Function
        int [[^foo]](int) {}
        int main() {
          [[foo]]([[foo]](42));
          auto *X = &[[foo]];
        }
      )cpp",

      R"cpp(// Function parameter in decl
        void foo(int [[^bar]]);
      )cpp",
      R"cpp(// Not touching any identifiers.
        struct Foo {
          [[~]]Foo() {};
        };
        void foo() {
          Foo f;
          f.[[^~]]Foo();
        }
      )cpp",
      R"cpp(// ObjC methods with split selectors.
        @interface Foo
          +(void) [[x]]:(int)a [[y]]:(int)b;
        @end
        @implementation Foo
          +(void) [[x]]:(int)a [[y]]:(int)b {}
        @end
        void go() {
          [Foo [[x]]:2 [[^y]]:4];
        }
      )cpp",
      R"cpp( // Label
        int main() {
          goto [[^theLabel]];
          [[theLabel]]:
            return 1;
        }
      )cpp",
  };
  for (const char *Test : Tests) {
    Annotations T(Test);
    auto TU = TestTU::withCode(T.code());
    TU.ExtraArgs.push_back("-xobjective-c++");
    auto AST = TU.build();
    EXPECT_THAT(findDocumentHighlights(AST, T.point()), highlightsFrom(T))
        << Test;
  }
}

TEST(HighlightsTest, ControlFlow) {
  const char *Tests[] = {
      R"cpp(
        // Highlight same-function returns.
        int fib(unsigned n) {
          if (n <= 1) [[ret^urn]] 1;
          [[return]] fib(n - 1) + fib(n - 2);

          // Returns from other functions not highlighted.
          auto Lambda = [] { return; };
          class LocalClass { void x() { return; } };
        }
      )cpp",

      R"cpp(
        #define FAIL() return false
        #define DO(x) { x; }
        bool foo(int n) {
          if (n < 0) [[FAIL]]();
          DO([[re^turn]] true)
        }
      )cpp",

      R"cpp(
        // Highlight loop control flow
        int magic() {
          int counter = 0;
          [[^for]] (char c : "fruit loops!") {
            if (c == ' ') [[continue]];
            counter += c;
            if (c == '!') [[break]];
            if (c == '?') [[return]] -1;
          }
          return counter;
        }
      )cpp",

      R"cpp(
        // Highlight loop and same-loop control flow
        void nonsense() {
          [[while]] (true) {
            if (false) [[bre^ak]];
            switch (1) break;
            [[continue]];
          }
        }
      )cpp",

      R"cpp(
        // Highlight switch for break (but not other breaks).
        void describe(unsigned n) {
          [[switch]](n) {
          case 0:
            break;
          [[default]]:
            [[^break]];
          }
        }
      )cpp",

      R"cpp(
        // Highlight case and exits for switch-break (but not other cases).
        void describe(unsigned n) {
          [[switch]](n) {
          case 0:
            break;
          [[case]] 1:
          [[default]]:
            [[return]];
            [[^break]];
          }
        }
      )cpp",

      R"cpp(
        // Highlight exits and switch for case
        void describe(unsigned n) {
          [[switch]](n) {
          case 0:
            break;
          [[case]] 1:
          [[d^efault]]:
            [[return]];
            [[break]];
          }
        }
      )cpp",

      R"cpp(
        // Highlight nothing for switch.
        void describe(unsigned n) {
          s^witch(n) {
          case 0:
            break;
          case 1:
          default:
            return;
            break;
          }
        }
      )cpp",

      R"cpp(
        // FIXME: match exception type against catch blocks
        int catchy() {
          try {                     // wrong: highlight try with matching catch
            try {                   // correct: has no matching catch
              [[thr^ow]] "oh no!";
            } catch (int) { }       // correct: catch doesn't match type
            [[return]] -1;          // correct: exits the matching catch
          } catch (const char*) { } // wrong: highlight matching catch
          [[return]] 42;            // wrong: throw doesn't exit function
        }
      )cpp",

      R"cpp(
        // Loop highlights goto exiting the loop, but not jumping within it.
        void jumpy() {
          [[wh^ile]](1) {
            up:
            if (0) [[goto]] out;
            goto up;
          }
          out: return;
        }
      )cpp",
  };
  for (const char *Test : Tests) {
    Annotations T(Test);
    auto TU = TestTU::withCode(T.code());
    TU.ExtraArgs.push_back("-fexceptions"); // FIXME: stop testing on PS4.
    auto AST = TU.build();
    EXPECT_THAT(findDocumentHighlights(AST, T.point()), highlightsFrom(T))
        << Test;
  }
}

MATCHER_P3(sym, Name, Decl, DefOrNone, "") {
  std::optional<Range> Def = DefOrNone;
  if (Name != arg.Name) {
    *result_listener << "Name is " << arg.Name;
    return false;
  }
  if (Decl != arg.PreferredDeclaration.range) {
    *result_listener << "Declaration is "
                     << llvm::to_string(arg.PreferredDeclaration);
    return false;
  }
  if (!Def && !arg.Definition)
    return true;
  if (Def && !arg.Definition) {
    *result_listener << "Has no definition";
    return false;
  }
  if (!Def && arg.Definition) {
    *result_listener << "Definition is " << llvm::to_string(*arg.Definition);
    return false;
  }
  if (arg.Definition->range != *Def) {
    *result_listener << "Definition is " << llvm::to_string(*arg.Definition);
    return false;
  }
  return true;
}

MATCHER_P(sym, Name, "") { return arg.Name == Name; }

MATCHER_P(rangeIs, R, "") { return arg.Loc.range == R; }
MATCHER_P(containerIs, C, "") {
  return arg.Loc.containerName.value_or("") == C;
}
MATCHER_P(attrsAre, A, "") { return arg.Attributes == A; }
MATCHER_P(hasID, ID, "") { return arg.ID == ID; }

TEST(LocateSymbol, WithIndex) {
  Annotations SymbolHeader(R"cpp(
        class $forward[[Forward]];
        class $foo[[Foo]] {};

        void $f1[[f1]]();

        inline void $f2[[f2]]() {}
      )cpp");
  Annotations SymbolCpp(R"cpp(
      class $forward[[forward]] {};
      void  $f1[[f1]]() {}
    )cpp");

  TestTU TU;
  TU.Code = std::string(SymbolCpp.code());
  TU.HeaderCode = std::string(SymbolHeader.code());
  auto Index = TU.index();
  auto LocateWithIndex = [&Index](const Annotations &Main) {
    auto AST = TestTU::withCode(Main.code()).build();
    return clangd::locateSymbolAt(AST, Main.point(), Index.get());
  };

  Annotations Test(R"cpp(// only declaration in AST.
        void [[f1]]();
        int main() {
          ^f1();
        }
      )cpp");
  EXPECT_THAT(LocateWithIndex(Test),
              ElementsAre(sym("f1", Test.range(), SymbolCpp.range("f1"))));

  Test = Annotations(R"cpp(// definition in AST.
        void [[f1]]() {}
        int main() {
          ^f1();
        }
      )cpp");
  EXPECT_THAT(LocateWithIndex(Test),
              ElementsAre(sym("f1", SymbolHeader.range("f1"), Test.range())));

  Test = Annotations(R"cpp(// forward declaration in AST.
        class [[Foo]];
        F^oo* create();
      )cpp");
  EXPECT_THAT(LocateWithIndex(Test),
              ElementsAre(sym("Foo", Test.range(), SymbolHeader.range("foo"))));

  Test = Annotations(R"cpp(// definition in AST.
        class [[Forward]] {};
        F^orward create();
      )cpp");
  EXPECT_THAT(
      LocateWithIndex(Test),
      ElementsAre(sym("Forward", SymbolHeader.range("forward"), Test.range())));
}

TEST(LocateSymbol, AnonymousStructFields) {
  auto Code = Annotations(R"cpp(
    struct $2[[Foo]] {
      struct { int $1[[x]]; };
      void foo() {
        // Make sure the implicit base is skipped.
        $1^x = 42;
      }
    };
    // Check that we don't skip explicit bases.
    int a = $2^Foo{}.x;
  )cpp");
  TestTU TU = TestTU::withCode(Code.code());
  auto AST = TU.build();
  EXPECT_THAT(locateSymbolAt(AST, Code.point("1"), TU.index().get()),
              UnorderedElementsAre(sym("x", Code.range("1"), Code.range("1"))));
  EXPECT_THAT(
      locateSymbolAt(AST, Code.point("2"), TU.index().get()),
      UnorderedElementsAre(sym("Foo", Code.range("2"), Code.range("2"))));
}

TEST(LocateSymbol, FindOverrides) {
  auto Code = Annotations(R"cpp(
    class Foo {
      virtual void $1[[fo^o]]() = 0;
    };
    class Bar : public Foo {
      void $2[[foo]]() override;
    };
  )cpp");
  TestTU TU = TestTU::withCode(Code.code());
  auto AST = TU.build();
  EXPECT_THAT(locateSymbolAt(AST, Code.point(), TU.index().get()),
              UnorderedElementsAre(sym("foo", Code.range("1"), std::nullopt),
                                   sym("foo", Code.range("2"), std::nullopt)));
}

TEST(LocateSymbol, WithIndexPreferredLocation) {
  Annotations SymbolHeader(R"cpp(
        class $p[[Proto]] {};
        void $f[[func]]() {};
      )cpp");
  TestTU TU;
  TU.HeaderCode = std::string(SymbolHeader.code());
  TU.HeaderFilename = "x.proto"; // Prefer locations in codegen files.
  auto Index = TU.index();

  Annotations Test(R"cpp(// only declaration in AST.
        // Shift to make range different.
        class Proto;
        void func() {}
        P$p^roto* create() {
          fu$f^nc();
          return nullptr;
        }
      )cpp");

  auto AST = TestTU::withCode(Test.code()).build();
  {
    auto Locs = clangd::locateSymbolAt(AST, Test.point("p"), Index.get());
    auto CodeGenLoc = SymbolHeader.range("p");
    EXPECT_THAT(Locs, ElementsAre(sym("Proto", CodeGenLoc, CodeGenLoc)));
  }
  {
    auto Locs = clangd::locateSymbolAt(AST, Test.point("f"), Index.get());
    auto CodeGenLoc = SymbolHeader.range("f");
    EXPECT_THAT(Locs, ElementsAre(sym("func", CodeGenLoc, CodeGenLoc)));
  }
}

TEST(LocateSymbol, All) {
  // Ranges in tests:
  //   $decl is the declaration location (if absent, no symbol is located)
  //   $def is the definition location (if absent, symbol has no definition)
  //   unnamed range becomes both $decl and $def.
  const char *Tests[] = {
      R"cpp(
        struct X {
          union {
            int [[a]];
            float b;
          };
        };
        int test(X &x) {
          return x.^a;
        }
      )cpp",

      R"cpp(// Local variable
        int main() {
          int [[bonjour]];
          ^bonjour = 2;
          int test1 = bonjour;
        }
      )cpp",

      R"cpp(// Struct
        namespace ns1 {
        struct [[MyClass]] {};
        } // namespace ns1
        int main() {
          ns1::My^Class* Params;
        }
      )cpp",

      R"cpp(// Function definition via pointer
        void [[foo]](int) {}
        int main() {
          auto *X = &^foo;
        }
      )cpp",

      R"cpp(// Function declaration via call
        int $decl[[foo]](int);
        int main() {
          return ^foo(42);
        }
      )cpp",

      R"cpp(// Field
        struct Foo { int [[x]]; };
        int main() {
          Foo bar;
          (void)bar.^x;
        }
      )cpp",

      R"cpp(// Field, member initializer
        struct Foo {
          int [[x]];
          Foo() : ^x(0) {}
        };
      )cpp",

      R"cpp(// Field, field designator
        struct Foo { int [[x]]; };
        int main() {
          Foo bar = { .^x = 2 };
        }
      )cpp",

      R"cpp(// Method call
        struct Foo { int $decl[[x]](); };
        int main() {
          Foo bar;
          bar.^x();
        }
      )cpp",

      R"cpp(// Typedef
        typedef int $decl[[Foo]];
        int main() {
          ^Foo bar;
        }
      )cpp",

      R"cpp(// Template type parameter
        template <typename [[T]]>
        void foo() { ^T t; }
      )cpp",

      R"cpp(// Template template type parameter
        template <template<typename> class [[T]]>
        void foo() { ^T<int> t; }
      )cpp",

      R"cpp(// Namespace
        namespace $decl[[ns]] {
        struct Foo { static void bar(); };
        } // namespace ns
        int main() { ^ns::Foo::bar(); }
      )cpp",

      R"cpp(// Macro
       class TTT { public: int a; };
       #define [[FF]](S) if (int b = S.a) {}
       void f() {
         TTT t;
         F^F(t);
       }
      )cpp",

      R"cpp(// Macro argument
       int [[i]];
       #define ADDRESSOF(X) &X;
       int *j = ADDRESSOF(^i);
      )cpp",
      R"cpp(// Macro argument appearing multiple times in expansion
        #define VALIDATE_TYPE(x) (void)x;
        #define ASSERT(expr)       \
          do {                     \
            VALIDATE_TYPE(expr);   \
            if (!expr);            \
          } while (false)
        bool [[waldo]]() { return true; }
        void foo() {
          ASSERT(wa^ldo());
        }
      )cpp",
      R"cpp(// Symbol concatenated inside macro (not supported)
       int *pi;
       #define POINTER(X) p ## X;
       int x = *POINTER(^i);
      )cpp",

      R"cpp(// Forward class declaration
        class $decl[[Foo]];
        class $def[[Foo]] {};
        F^oo* foo();
      )cpp",

      R"cpp(// Function declaration
        void $decl[[foo]]();
        void g() { f^oo(); }
        void $def[[foo]]() {}
      )cpp",

      R"cpp(
        #define FF(name) class name##_Test {};
        [[FF]](my);
        void f() { my^_Test a; }
      )cpp",

      R"cpp(
         #define FF() class [[Test]] {};
         FF();
         void f() { T^est a; }
      )cpp",

      R"cpp(// explicit template specialization
        template <typename T>
        struct Foo { void bar() {} };

        template <>
        struct [[Foo]]<int> { void bar() {} };

        void foo() {
          Foo<char> abc;
          Fo^o<int> b;
        }
      )cpp",

      R"cpp(// implicit template specialization
        template <typename T>
        struct [[Foo]] { void bar() {} };
        template <>
        struct Foo<int> { void bar() {} };
        void foo() {
          Fo^o<char> abc;
          Foo<int> b;
        }
      )cpp",

      R"cpp(// partial template specialization
        template <typename T>
        struct Foo { void bar() {} };
        template <typename T>
        struct [[Foo]]<T*> { void bar() {} };
        ^Foo<int*> x;
      )cpp",

      R"cpp(// function template specializations
        template <class T>
        void foo(T) {}
        template <>
        void [[foo]](int) {}
        void bar() {
          fo^o(10);
        }
      )cpp",

      R"cpp(// variable template decls
        template <class T>
        T var = T();

        template <>
        double [[var]]<int> = 10;

        double y = va^r<int>;
      )cpp",

      R"cpp(// No implicit constructors
        struct X {
          X(X&& x) = default;
        };
        X $decl[[makeX]]();
        void foo() {
          auto x = m^akeX();
        }
      )cpp",

      R"cpp(
        struct X {
          X& $decl[[operator]]++();
        };
        void foo(X& x) {
          +^+x;
        }
      )cpp",

      R"cpp(
        struct S1 { void f(); };
        struct S2 { S1 * $decl[[operator]]->(); };
        void test(S2 s2) {
          s2-^>f();
        }
      )cpp",

      R"cpp(// Declaration of explicit template specialization
        template <typename T>
        struct $decl[[$def[[Foo]]]] {};

        template <>
        struct Fo^o<int> {};
      )cpp",

      R"cpp(// Declaration of partial template specialization
        template <typename T>
        struct $decl[[$def[[Foo]]]] {};

        template <typename T>
        struct Fo^o<T*> {};
      )cpp",

      R"cpp(// Definition on ClassTemplateDecl
        namespace ns {
          // Forward declaration.
          template<typename T>
          struct $decl[[Foo]];

          template <typename T>
          struct $def[[Foo]] {};
        }

        using ::ns::Fo^o;
      )cpp",

      R"cpp(// auto builtin type (not supported)
        ^auto x = 42;
      )cpp",

      R"cpp(// auto on lambda
        auto x = [[[]]]{};
        ^auto y = x;
      )cpp",

      R"cpp(// auto on struct
        namespace ns1 {
        struct [[S1]] {};
        } // namespace ns1

        ^auto x = ns1::S1{};
      )cpp",

      R"cpp(// decltype on struct
        namespace ns1 {
        struct [[S1]] {};
        } // namespace ns1

        ns1::S1 i;
        ^decltype(i) j;
      )cpp",

      R"cpp(// decltype(auto) on struct
        namespace ns1 {
        struct [[S1]] {};
        } // namespace ns1

        ns1::S1 i;
        ns1::S1& j = i;
        ^decltype(auto) k = j;
      )cpp",

      R"cpp(// auto on template class
        template<typename T> class [[Foo]] {};

        ^auto x = Foo<int>();
      )cpp",

      R"cpp(// auto on template class with forward declared class
        template<typename T> class [[Foo]] {};
        class X;

        ^auto x = Foo<X>();
      )cpp",

      R"cpp(// auto on specialized template class
        template<typename T> class Foo {};
        template<> class [[Foo]]<int> {};

        ^auto x = Foo<int>();
      )cpp",

      R"cpp(// auto on initializer list.
        namespace std
        {
          template<class _E>
          class [[initializer_list]] {};
        }

        ^auto i = {1,2};
      )cpp",

      R"cpp(// auto function return with trailing type
        struct [[Bar]] {};
        ^auto test() -> decltype(Bar()) {
          return Bar();
        }
      )cpp",

      R"cpp(// decltype in trailing return type
        struct [[Bar]] {};
        auto test() -> ^decltype(Bar()) {
          return Bar();
        }
      )cpp",

      R"cpp(// auto in function return
        struct [[Bar]] {};
        ^auto test() {
          return Bar();
        }
      )cpp",

      R"cpp(// auto& in function return
        struct [[Bar]] {};
        ^auto& test() {
          static Bar x;
          return x;
        }
      )cpp",

      R"cpp(// auto* in function return
        struct [[Bar]] {};
        ^auto* test() {
          Bar* x;
          return x;
        }
      )cpp",

      R"cpp(// const auto& in function return
        struct [[Bar]] {};
        const ^auto& test() {
          static Bar x;
          return x;
        }
      )cpp",

      R"cpp(// auto lambda param where there's a single instantiation
        struct [[Bar]] {};
        auto Lambda = [](^auto){ return 0; };
        int x = Lambda(Bar{});
      )cpp",

      R"cpp(// decltype(auto) in function return
        struct [[Bar]] {};
        ^decltype(auto) test() {
          return Bar();
        }
      )cpp",

      R"cpp(// decltype of function with trailing return type.
        struct [[Bar]] {};
        auto test() -> decltype(Bar()) {
          return Bar();
        }
        void foo() {
          ^decltype(test()) i = test();
        }
      )cpp",

      R"cpp(// Override specifier jumps to overridden method
        class Y { virtual void $decl[[a]]() = 0; };
        class X : Y { void a() ^override {} };
      )cpp",

      R"cpp(// Final specifier jumps to overridden method
        class Y { virtual void $decl[[a]]() = 0; };
        class X : Y { void a() ^final {} };
      )cpp",

      R"cpp(// Heuristic resolution of dependent method
        template <typename T>
        struct S {
          void [[bar]]() {}
        };

        template <typename T>
        void foo(S<T> arg) {
          arg.ba^r();
        }
      )cpp",

      R"cpp(// Heuristic resolution of dependent method via this->
        template <typename T>
        struct S {
          void [[foo]]() {
            this->fo^o();
          }
        };
      )cpp",

      R"cpp(// Heuristic resolution of dependent static method
        template <typename T>
        struct S {
          static void [[bar]]() {}
        };

        template <typename T>
        void foo() {
          S<T>::ba^r();
        }
      )cpp",

      R"cpp(// Heuristic resolution of dependent method
            // invoked via smart pointer
        template <typename> struct S { void [[foo]]() {} };
        template <typename T> struct unique_ptr {
          T* operator->();
        };
        template <typename T>
        void test(unique_ptr<S<T>>& V) {
          V->fo^o();
        }
      )cpp",

      R"cpp(// Heuristic resolution of dependent enumerator
        template <typename T>
        struct Foo {
          enum class E { [[A]], B };
          E e = E::A^;
        };
      )cpp",

      R"cpp(// Enum base
        typedef int $decl[[MyTypeDef]];
        enum Foo : My^TypeDef {};
      )cpp",
      R"cpp(// Enum base
        typedef int $decl[[MyTypeDef]];
        enum Foo : My^TypeDef;
      )cpp",
      R"cpp(// Enum base
        using $decl[[MyTypeDef]] = int;
        enum Foo : My^TypeDef {};
      )cpp",

      R"objc(
        @protocol Dog;
        @protocol $decl[[Dog]]
        - (void)bark;
        @end
        id<Do^g> getDoggo() {
          return 0;
        }
      )objc",

      R"objc(
        @interface Cat
        @end
        @implementation Cat
        @end
        @interface $decl[[Cat]] (Exte^nsion)
        - (void)meow;
        @end
        @implementation $def[[Cat]] (Extension)
        - (void)meow {}
        @end
      )objc",

      R"objc(
        @class $decl[[Foo]];
        Fo^o * getFoo() {
          return 0;
        }
      )objc",

      R"objc(// Prefer interface definition over forward declaration
        @class Foo;
        @interface $decl[[Foo]]
        @end
        Fo^o * getFoo() {
          return 0;
        }
      )objc",

      R"objc(
        @class Foo;
        @interface $decl[[Foo]]
        @end
        @implementation $def[[Foo]]
        @end
        Fo^o * getFoo() {
          return 0;
        }
      )objc",

      R"objc(// Method decl and definition for ObjC class.
        @interface Cat
        - (void)$decl[[meow]];
        @end
        @implementation Cat
        - (void)$def[[meow]] {}
        @end
        void makeNoise(Cat *kitty) {
          [kitty me^ow];
        }
      )objc",

      R"objc(// Method decl and definition for ObjC category.
        @interface Dog
        @end
        @interface Dog (Play)
        - (void)$decl[[runAround]];
        @end
        @implementation Dog (Play)
        - (void)$def[[runAround]] {}
        @end
        void play(Dog *dog) {
          [dog run^Around];
        }
      )objc",

      R"objc(// Method decl and definition for ObjC class extension.
        @interface Dog
        @end
        @interface Dog ()
        - (void)$decl[[howl]];
        @end
        @implementation Dog
        - (void)$def[[howl]] {}
        @end
        void play(Dog *dog) {
          [dog ho^wl];
        }
      )objc",
      R"cpp(
        struct PointerIntPairInfo {
          static void *getPointer(void *Value);
        };

        template <typename Info = PointerIntPairInfo> struct PointerIntPair {
          void *Value;
          void *getPointer() const { return Info::get^Pointer(Value); }
        };
    )cpp"};
  for (const char *Test : Tests) {
    Annotations T(Test);
    std::optional<Range> WantDecl;
    std::optional<Range> WantDef;
    if (!T.ranges().empty())
      WantDecl = WantDef = T.range();
    if (!T.ranges("decl").empty())
      WantDecl = T.range("decl");
    if (!T.ranges("def").empty())
      WantDef = T.range("def");

    TestTU TU;
    TU.Code = std::string(T.code());

    TU.ExtraArgs.push_back("-xobjective-c++");

    auto AST = TU.build();
    auto Results = locateSymbolAt(AST, T.point());

    if (!WantDecl) {
      EXPECT_THAT(Results, IsEmpty()) << Test;
    } else {
      ASSERT_THAT(Results, ::testing::SizeIs(1)) << Test;
      EXPECT_EQ(Results[0].PreferredDeclaration.range, *WantDecl) << Test;
      EXPECT_TRUE(Results[0].ID) << Test;
      std::optional<Range> GotDef;
      if (Results[0].Definition)
        GotDef = Results[0].Definition->range;
      EXPECT_EQ(WantDef, GotDef) << Test;
    }
  }
}
TEST(LocateSymbol, ValidSymbolID) {
  auto T = Annotations(R"cpp(
    #define MACRO(x, y) ((x) + (y))
    int add(int x, int y) { return $MACRO^MACRO(x, y); }
    int sum = $add^add(1, 2);
  )cpp");

  TestTU TU = TestTU::withCode(T.code());
  auto AST = TU.build();
  auto Index = TU.index();
  EXPECT_THAT(locateSymbolAt(AST, T.point("add"), Index.get()),
              ElementsAre(AllOf(sym("add"),
                                hasID(getSymbolID(&findDecl(AST, "add"))))));
  EXPECT_THAT(
      locateSymbolAt(AST, T.point("MACRO"), Index.get()),
      ElementsAre(AllOf(sym("MACRO"),
                        hasID(findSymbol(TU.headerSymbols(), "MACRO").ID))));
}

TEST(LocateSymbol, AllMulti) {
  // Ranges in tests:
  //   $declN is the declaration location
  //   $defN is the definition location (if absent, symbol has no definition)
  //
  // NOTE:
  //   N starts at 0.
  struct ExpectedRanges {
    Range WantDecl;
    std::optional<Range> WantDef;
  };
  const char *Tests[] = {
      R"objc(
        @interface $decl0[[Cat]]
        @end
        @implementation $def0[[Cat]]
        @end
        @interface $decl1[[Ca^t]] (Extension)
        - (void)meow;
        @end
        @implementation $def1[[Cat]] (Extension)
        - (void)meow {}
        @end
      )objc",

      R"objc(
        @interface $decl0[[Cat]]
        @end
        @implementation $def0[[Cat]]
        @end
        @interface $decl1[[Cat]] (Extension)
        - (void)meow;
        @end
        @implementation $def1[[Ca^t]] (Extension)
        - (void)meow {}
        @end
      )objc",

      R"objc(
        @interface $decl0[[Cat]]
        @end
        @interface $decl1[[Ca^t]] ()
        - (void)meow;
        @end
        @implementation $def0[[$def1[[Cat]]]]
        - (void)meow {}
        @end
      )objc",
  };
  for (const char *Test : Tests) {
    Annotations T(Test);
    std::vector<ExpectedRanges> Ranges;
    for (int Idx = 0; true; Idx++) {
      bool HasDecl = !T.ranges("decl" + std::to_string(Idx)).empty();
      bool HasDef = !T.ranges("def" + std::to_string(Idx)).empty();
      if (!HasDecl && !HasDef)
        break;
      ExpectedRanges Range;
      if (HasDecl)
        Range.WantDecl = T.range("decl" + std::to_string(Idx));
      if (HasDef)
        Range.WantDef = T.range("def" + std::to_string(Idx));
      Ranges.push_back(Range);
    }

    TestTU TU;
    TU.Code = std::string(T.code());
    TU.ExtraArgs.push_back("-xobjective-c++");

    auto AST = TU.build();
    auto Results = locateSymbolAt(AST, T.point());

    ASSERT_THAT(Results, ::testing::SizeIs(Ranges.size())) << Test;
    for (size_t Idx = 0; Idx < Ranges.size(); Idx++) {
      EXPECT_EQ(Results[Idx].PreferredDeclaration.range, Ranges[Idx].WantDecl)
          << "($decl" << Idx << ")" << Test;
      std::optional<Range> GotDef;
      if (Results[Idx].Definition)
        GotDef = Results[Idx].Definition->range;
      EXPECT_EQ(GotDef, Ranges[Idx].WantDef) << "($def" << Idx << ")" << Test;
    }
  }
}

// LocateSymbol test cases that produce warnings.
// These are separated out from All so that in All we can assert
// that there are no diagnostics.
TEST(LocateSymbol, Warnings) {
  const char *Tests[] = {
      R"cpp(// Field, GNU old-style field designator
        struct Foo { int [[x]]; };
        int main() {
          Foo bar = { ^x : 1 };
        }
      )cpp",

      R"cpp(// Macro
        #define MACRO 0
        #define [[MACRO]] 1
        int main() { return ^MACRO; }
        #define MACRO 2
        #undef macro
      )cpp",
  };

  for (const char *Test : Tests) {
    Annotations T(Test);
    std::optional<Range> WantDecl;
    std::optional<Range> WantDef;
    if (!T.ranges().empty())
      WantDecl = WantDef = T.range();
    if (!T.ranges("decl").empty())
      WantDecl = T.range("decl");
    if (!T.ranges("def").empty())
      WantDef = T.range("def");

    TestTU TU;
    TU.Code = std::string(T.code());

    auto AST = TU.build();
    auto Results = locateSymbolAt(AST, T.point());

    if (!WantDecl) {
      EXPECT_THAT(Results, IsEmpty()) << Test;
    } else {
      ASSERT_THAT(Results, ::testing::SizeIs(1)) << Test;
      EXPECT_EQ(Results[0].PreferredDeclaration.range, *WantDecl) << Test;
      std::optional<Range> GotDef;
      if (Results[0].Definition)
        GotDef = Results[0].Definition->range;
      EXPECT_EQ(WantDef, GotDef) << Test;
    }
  }
}

TEST(LocateSymbol, TextualSmoke) {
  auto T = Annotations(
      R"cpp(
        struct [[MyClass]] {};
        // Comment mentioning M^yClass
      )cpp");

  auto TU = TestTU::withCode(T.code());
  auto AST = TU.build();
  auto Index = TU.index();
  EXPECT_THAT(
      locateSymbolAt(AST, T.point(), Index.get()),
      ElementsAre(AllOf(sym("MyClass", T.range(), T.range()),
                        hasID(getSymbolID(&findDecl(AST, "MyClass"))))));
}

TEST(LocateSymbol, Textual) {
  const char *Tests[] = {
      R"cpp(// Comment
        struct [[MyClass]] {};
        // Comment mentioning M^yClass
      )cpp",
      R"cpp(// String
        struct MyClass {};
        // Not triggered for string literal tokens.
        const char* s = "String literal mentioning M^yClass";
      )cpp",
      R"cpp(// Ifdef'ed out code
        struct [[MyClass]] {};
        #ifdef WALDO
          M^yClass var;
        #endif
      )cpp",
      R"cpp(// Macro definition
        struct [[MyClass]] {};
        #define DECLARE_MYCLASS_OBJ(name) M^yClass name;
      )cpp",
      R"cpp(// Invalid code
        /*error-ok*/
        int myFunction(int);
        // Not triggered for token which survived preprocessing.
        int var = m^yFunction();
      )cpp"};

  for (const char *Test : Tests) {
    Annotations T(Test);
    std::optional<Range> WantDecl;
    if (!T.ranges().empty())
      WantDecl = T.range();

    auto TU = TestTU::withCode(T.code());

    auto AST = TU.build();
    auto Index = TU.index();
    auto Word = SpelledWord::touching(
        cantFail(sourceLocationInMainFile(AST.getSourceManager(), T.point())),
        AST.getTokens(), AST.getLangOpts());
    if (!Word) {
      ADD_FAILURE() << "No word touching point!" << Test;
      continue;
    }
    auto Results = locateSymbolTextually(*Word, AST, Index.get(),
                                         testPath(TU.Filename), ASTNodeKind());

    if (!WantDecl) {
      EXPECT_THAT(Results, IsEmpty()) << Test;
    } else {
      ASSERT_THAT(Results, ::testing::SizeIs(1)) << Test;
      EXPECT_EQ(Results[0].PreferredDeclaration.range, *WantDecl) << Test;
    }
  }
} // namespace

TEST(LocateSymbol, Ambiguous) {
  auto T = Annotations(R"cpp(
    struct Foo {
      Foo();
      Foo(Foo&&);
      $ConstructorLoc[[Foo]](const char*);
    };

    Foo f();

    void g(Foo foo);

    void call() {
      const char* str = "123";
      Foo a = $1^str;
      Foo b = Foo($2^str);
      Foo c = $3^f();
      $4^g($5^f());
      g($6^str);
      Foo ab$7^c;
      Foo ab$8^cd("asdf");
      Foo foox = Fo$9^o("asdf");
      Foo abcde$10^("asdf");
      Foo foox2 = Foo$11^("asdf");
    }

    template <typename T>
    struct S {
      void $NonstaticOverload1[[bar]](int);
      void $NonstaticOverload2[[bar]](float);

      static void $StaticOverload1[[baz]](int);
      static void $StaticOverload2[[baz]](float);
    };

    template <typename T, typename U>
    void dependent_call(S<T> s, U u) {
      s.ba$12^r(u);
      S<T>::ba$13^z(u);
    }
  )cpp");
  auto TU = TestTU::withCode(T.code());
  // FIXME: Go-to-definition in a template requires disabling delayed template
  // parsing.
  TU.ExtraArgs.push_back("-fno-delayed-template-parsing");
  auto AST = TU.build();
  // Ordered assertions are deliberate: we expect a predictable order.
  EXPECT_THAT(locateSymbolAt(AST, T.point("1")), ElementsAre(sym("str")));
  EXPECT_THAT(locateSymbolAt(AST, T.point("2")), ElementsAre(sym("str")));
  EXPECT_THAT(locateSymbolAt(AST, T.point("3")), ElementsAre(sym("f")));
  EXPECT_THAT(locateSymbolAt(AST, T.point("4")), ElementsAre(sym("g")));
  EXPECT_THAT(locateSymbolAt(AST, T.point("5")), ElementsAre(sym("f")));
  EXPECT_THAT(locateSymbolAt(AST, T.point("6")), ElementsAre(sym("str")));
  // FIXME: Target the constructor as well.
  EXPECT_THAT(locateSymbolAt(AST, T.point("7")), ElementsAre(sym("abc")));
  // FIXME: Target the constructor as well.
  EXPECT_THAT(locateSymbolAt(AST, T.point("8")), ElementsAre(sym("abcd")));
  // FIXME: Target the constructor as well.
  EXPECT_THAT(locateSymbolAt(AST, T.point("9")), ElementsAre(sym("Foo")));
  EXPECT_THAT(locateSymbolAt(AST, T.point("10")),
              ElementsAre(sym("Foo", T.range("ConstructorLoc"), std::nullopt)));
  EXPECT_THAT(locateSymbolAt(AST, T.point("11")),
              ElementsAre(sym("Foo", T.range("ConstructorLoc"), std::nullopt)));
  // These assertions are unordered because the order comes from
  // CXXRecordDecl::lookupDependentName() which doesn't appear to provide
  // an order guarantee.
  EXPECT_THAT(locateSymbolAt(AST, T.point("12")),
              UnorderedElementsAre(
                  sym("bar", T.range("NonstaticOverload1"), std::nullopt),
                  sym("bar", T.range("NonstaticOverload2"), std::nullopt)));
  EXPECT_THAT(locateSymbolAt(AST, T.point("13")),
              UnorderedElementsAre(
                  sym("baz", T.range("StaticOverload1"), std::nullopt),
                  sym("baz", T.range("StaticOverload2"), std::nullopt)));
}

TEST(LocateSymbol, TextualDependent) {
  // Put the declarations in the header to make sure we are
  // finding them via the index heuristic and not the
  // nearby-ident heuristic.
  Annotations Header(R"cpp(
        struct Foo {
          void $FooLoc[[uniqueMethodName]]();
        };
        struct Bar {
          void $BarLoc[[uniqueMethodName]]();
        };
        )cpp");
  Annotations Source(R"cpp(
        template <typename T>
        void f(T t) {
          t.u^niqueMethodName();
        }
      )cpp");
  TestTU TU;
  TU.Code = std::string(Source.code());
  TU.HeaderCode = std::string(Header.code());
  auto AST = TU.build();
  auto Index = TU.index();
  // Need to use locateSymbolAt() since we are testing an
  // interaction between locateASTReferent() and
  // locateSymbolNamedTextuallyAt().
  auto Results = locateSymbolAt(AST, Source.point(), Index.get());
  EXPECT_THAT(
      Results,
      UnorderedElementsAre(
          sym("uniqueMethodName", Header.range("FooLoc"), std::nullopt),
          sym("uniqueMethodName", Header.range("BarLoc"), std::nullopt)));
}

TEST(LocateSymbol, Alias) {
  const char *Tests[] = {
      R"cpp(
      template <class T> struct function {};
      template <class T> using [[callback]] = function<T()>;

      c^allback<int> foo;
    )cpp",

      // triggered on non-definition of a renaming alias: should not give any
      // underlying decls.
      R"cpp(
      class Foo {};
      typedef Foo [[Bar]];

      B^ar b;
    )cpp",
      R"cpp(
      class Foo {};
      using [[Bar]] = Foo; // definition
      Ba^r b;
    )cpp",

      // triggered on the underlying decl of a renaming alias.
      R"cpp(
      class [[Foo]];
      using Bar = Fo^o;
    )cpp",

      // triggered on definition of a non-renaming alias: should give underlying
      // decls.
      R"cpp(
      namespace ns { class [[Foo]] {}; }
      using ns::F^oo;
    )cpp",

      R"cpp(
      namespace ns { int [[x]](char); int [[x]](double); }
      using ns::^x;
    )cpp",

      R"cpp(
      namespace ns { int [[x]](char); int x(double); }
      using ns::[[x]];
      int y = ^x('a');
    )cpp",

      R"cpp(
      namespace ns { class [[Foo]] {}; }
      using ns::[[Foo]];
      F^oo f;
    )cpp",

      // other cases that don't matter much.
      R"cpp(
      class Foo {};
      typedef Foo [[Ba^r]];
    )cpp",
      R"cpp(
      class Foo {};
      using [[B^ar]] = Foo;
    )cpp",

      // Member of dependent base
      R"cpp(
      template <typename T>
      struct Base {
        void [[waldo]]() {}
      };
      template <typename T>
      struct Derived : Base<T> {
        using Base<T>::w^aldo;
      };
    )cpp",
  };

  for (const auto *Case : Tests) {
    SCOPED_TRACE(Case);
    auto T = Annotations(Case);
    auto AST = TestTU::withCode(T.code()).build();
    EXPECT_THAT(locateSymbolAt(AST, T.point()),
                UnorderedPointwise(declRange(), T.ranges()));
  }
}

TEST(LocateSymbol, RelPathsInCompileCommand) {
  // The source is in "/clangd-test/src".
  // We build in "/clangd-test/build".

  Annotations SourceAnnotations(R"cpp(
#include "header_in_preamble.h"
int [[foo]];
#include "header_not_in_preamble.h"
int baz = f$p1^oo + bar_pre$p2^amble + bar_not_pre$p3^amble;
)cpp");

  Annotations HeaderInPreambleAnnotations(R"cpp(
int [[bar_preamble]];
)cpp");

  Annotations HeaderNotInPreambleAnnotations(R"cpp(
int [[bar_not_preamble]];
)cpp");

  // Make the compilation paths appear as ../src/foo.cpp in the compile
  // commands.
  SmallString<32> RelPathPrefix("..");
  llvm::sys::path::append(RelPathPrefix, "src");
  std::string BuildDir = testPath("build");
  MockCompilationDatabase CDB(BuildDir, RelPathPrefix);

  MockFS FS;
  ClangdServer Server(CDB, FS, ClangdServer::optsForTest());

  // Fill the filesystem.
  auto FooCpp = testPath("src/foo.cpp");
  FS.Files[FooCpp] = "";
  auto HeaderInPreambleH = testPath("src/header_in_preamble.h");
  FS.Files[HeaderInPreambleH] = std::string(HeaderInPreambleAnnotations.code());
  auto HeaderNotInPreambleH = testPath("src/header_not_in_preamble.h");
  FS.Files[HeaderNotInPreambleH] =
      std::string(HeaderNotInPreambleAnnotations.code());

  runAddDocument(Server, FooCpp, SourceAnnotations.code());

  // Go to a definition in main source file.
  auto Locations =
      runLocateSymbolAt(Server, FooCpp, SourceAnnotations.point("p1"));
  EXPECT_TRUE(bool(Locations)) << "findDefinitions returned an error";
  EXPECT_THAT(*Locations, ElementsAre(sym("foo", SourceAnnotations.range(),
                                          SourceAnnotations.range())));

  // Go to a definition in header_in_preamble.h.
  Locations = runLocateSymbolAt(Server, FooCpp, SourceAnnotations.point("p2"));
  EXPECT_TRUE(bool(Locations)) << "findDefinitions returned an error";
  EXPECT_THAT(
      *Locations,
      ElementsAre(sym("bar_preamble", HeaderInPreambleAnnotations.range(),
                      HeaderInPreambleAnnotations.range())));

  // Go to a definition in header_not_in_preamble.h.
  Locations = runLocateSymbolAt(Server, FooCpp, SourceAnnotations.point("p3"));
  EXPECT_TRUE(bool(Locations)) << "findDefinitions returned an error";
  EXPECT_THAT(*Locations,
              ElementsAre(sym("bar_not_preamble",
                              HeaderNotInPreambleAnnotations.range(),
                              HeaderNotInPreambleAnnotations.range())));
}

TEST(GoToInclude, All) {
  MockFS FS;
  MockCompilationDatabase CDB;
  ClangdServer Server(CDB, FS, ClangdServer::optsForTest());

  auto FooCpp = testPath("foo.cpp");
  const char *SourceContents = R"cpp(
  #include ^"$2^foo.h$3^"
  #include "$4^invalid.h"
  int b = a;
  // test
  int foo;
  #in$5^clude "$6^foo.h"$7^
  )cpp";
  Annotations SourceAnnotations(SourceContents);
  FS.Files[FooCpp] = std::string(SourceAnnotations.code());
  auto FooH = testPath("foo.h");

  const char *HeaderContents = R"cpp([[]]#pragma once
                                     int a;
                                     )cpp";
  Annotations HeaderAnnotations(HeaderContents);
  FS.Files[FooH] = std::string(HeaderAnnotations.code());

  runAddDocument(Server, FooH, HeaderAnnotations.code());
  runAddDocument(Server, FooCpp, SourceAnnotations.code());

  // Test include in preamble.
  auto Locations = runLocateSymbolAt(Server, FooCpp, SourceAnnotations.point());
  ASSERT_TRUE(bool(Locations)) << "locateSymbolAt returned an error";
  EXPECT_THAT(*Locations, ElementsAre(sym("foo.h", HeaderAnnotations.range(),
                                          HeaderAnnotations.range())));

  // Test include in preamble, last char.
  Locations = runLocateSymbolAt(Server, FooCpp, SourceAnnotations.point("2"));
  ASSERT_TRUE(bool(Locations)) << "locateSymbolAt returned an error";
  EXPECT_THAT(*Locations, ElementsAre(sym("foo.h", HeaderAnnotations.range(),
                                          HeaderAnnotations.range())));

  Locations = runLocateSymbolAt(Server, FooCpp, SourceAnnotations.point("3"));
  ASSERT_TRUE(bool(Locations)) << "locateSymbolAt returned an error";
  EXPECT_THAT(*Locations, ElementsAre(sym("foo.h", HeaderAnnotations.range(),
                                          HeaderAnnotations.range())));

  // Test include outside of preamble.
  Locations = runLocateSymbolAt(Server, FooCpp, SourceAnnotations.point("6"));
  ASSERT_TRUE(bool(Locations)) << "locateSymbolAt returned an error";
  EXPECT_THAT(*Locations, ElementsAre(sym("foo.h", HeaderAnnotations.range(),
                                          HeaderAnnotations.range())));

  // Test a few positions that do not result in Locations.
  Locations = runLocateSymbolAt(Server, FooCpp, SourceAnnotations.point("4"));
  ASSERT_TRUE(bool(Locations)) << "locateSymbolAt returned an error";
  EXPECT_THAT(*Locations, IsEmpty());

  Locations = runLocateSymbolAt(Server, FooCpp, SourceAnnotations.point("5"));
  ASSERT_TRUE(bool(Locations)) << "locateSymbolAt returned an error";
  EXPECT_THAT(*Locations, ElementsAre(sym("foo.h", HeaderAnnotations.range(),
                                          HeaderAnnotations.range())));

  Locations = runLocateSymbolAt(Server, FooCpp, SourceAnnotations.point("7"));
  ASSERT_TRUE(bool(Locations)) << "locateSymbolAt returned an error";
  EXPECT_THAT(*Locations, ElementsAre(sym("foo.h", HeaderAnnotations.range(),
                                          HeaderAnnotations.range())));

  // Objective C #import directive.
  Annotations ObjC(R"objc(
  #import "^foo.h"
  )objc");
  auto FooM = testPath("foo.m");
  FS.Files[FooM] = std::string(ObjC.code());

  runAddDocument(Server, FooM, ObjC.code());
  Locations = runLocateSymbolAt(Server, FooM, ObjC.point());
  ASSERT_TRUE(bool(Locations)) << "locateSymbolAt returned an error";
  EXPECT_THAT(*Locations, ElementsAre(sym("foo.h", HeaderAnnotations.range(),
                                          HeaderAnnotations.range())));
}

TEST(LocateSymbol, WithPreamble) {
  // Test stragety: AST should always use the latest preamble instead of last
  // good preamble.
  MockFS FS;
  MockCompilationDatabase CDB;
  ClangdServer Server(CDB, FS, ClangdServer::optsForTest());

  auto FooCpp = testPath("foo.cpp");
  // The trigger locations must be the same.
  Annotations FooWithHeader(R"cpp(#include "fo^o.h")cpp");
  Annotations FooWithoutHeader(R"cpp(double    [[fo^o]]();)cpp");

  FS.Files[FooCpp] = std::string(FooWithHeader.code());

  auto FooH = testPath("foo.h");
  Annotations FooHeader(R"cpp([[]])cpp");
  FS.Files[FooH] = std::string(FooHeader.code());

  runAddDocument(Server, FooCpp, FooWithHeader.code());
  // LocateSymbol goes to a #include file: the result comes from the preamble.
  EXPECT_THAT(
      cantFail(runLocateSymbolAt(Server, FooCpp, FooWithHeader.point())),
      ElementsAre(sym("foo.h", FooHeader.range(), FooHeader.range())));

  // Only preamble is built, and no AST is built in this request.
  Server.addDocument(FooCpp, FooWithoutHeader.code(), "null",
                     WantDiagnostics::No);
  // We build AST here, and it should use the latest preamble rather than the
  // stale one.
  EXPECT_THAT(
      cantFail(runLocateSymbolAt(Server, FooCpp, FooWithoutHeader.point())),
      ElementsAre(sym("foo", FooWithoutHeader.range(), std::nullopt)));

  // Reset test environment.
  runAddDocument(Server, FooCpp, FooWithHeader.code());
  // Both preamble and AST are built in this request.
  Server.addDocument(FooCpp, FooWithoutHeader.code(), "null",
                     WantDiagnostics::Yes);
  // Use the AST being built in above request.
  EXPECT_THAT(
      cantFail(runLocateSymbolAt(Server, FooCpp, FooWithoutHeader.point())),
      ElementsAre(sym("foo", FooWithoutHeader.range(), std::nullopt)));
}

TEST(LocateSymbol, NearbyTokenSmoke) {
  auto T = Annotations(R"cpp(
    // prints e^rr and crashes
    void die(const char* [[err]]);
  )cpp");
  auto AST = TestTU::withCode(T.code()).build();
  // We don't pass an index, so can't hit index-based fallback.
  EXPECT_THAT(locateSymbolAt(AST, T.point()),
              ElementsAre(sym("err", T.range(), T.range())));
}

TEST(LocateSymbol, NearbyIdentifier) {
  const char *Tests[] = {
      R"cpp(
      // regular identifiers (won't trigger)
      int hello;
      int y = he^llo;
    )cpp",
      R"cpp(
      // disabled preprocessor sections
      int [[hello]];
      #if 0
      int y = ^hello;
      #endif
    )cpp",
      R"cpp(
      // comments
      // he^llo, world
      int [[hello]];
    )cpp",
      R"cpp(
      // not triggered by string literals
      int hello;
      const char* greeting = "h^ello, world";
    )cpp",

      R"cpp(
      // can refer to macro invocations
      #define INT int
      [[INT]] x;
      // I^NT
    )cpp",

      R"cpp(
      // can refer to macro invocations (even if they expand to nothing)
      #define EMPTY
      [[EMPTY]] int x;
      // E^MPTY
    )cpp",

      R"cpp(
      // prefer nearest occurrence, backwards is worse than forwards
      int hello;
      int x = hello;
      // h^ello
      int y = [[hello]];
      int z = hello;
    )cpp",

      R"cpp(
      // short identifiers find near results
      int [[hi]];
      // h^i
    )cpp",
      R"cpp(
      // short identifiers don't find far results
      int hi;



      // h^i




      int x = hi;
    )cpp",
      R"cpp(
      // prefer nearest occurrence even if several matched tokens
      // have the same value of `floor(log2(<token line> - <word line>))`.
      int hello;
      int x = hello, y = hello;
      int z = [[hello]];
      // h^ello
    )cpp"};
  for (const char *Test : Tests) {
    Annotations T(Test);
    auto AST = TestTU::withCode(T.code()).build();
    const auto &SM = AST.getSourceManager();
    std::optional<Range> Nearby;
    auto Word =
        SpelledWord::touching(cantFail(sourceLocationInMainFile(SM, T.point())),
                              AST.getTokens(), AST.getLangOpts());
    if (!Word) {
      ADD_FAILURE() << "No word at point! " << Test;
      continue;
    }
    if (const auto *Tok = findNearbyIdentifier(*Word, AST.getTokens()))
      Nearby = halfOpenToRange(SM, CharSourceRange::getCharRange(
                                       Tok->location(), Tok->endLocation()));
    if (T.ranges().empty())
      EXPECT_THAT(Nearby, Eq(std::nullopt)) << Test;
    else
      EXPECT_EQ(Nearby, T.range()) << Test;
  }
}

TEST(FindImplementations, Inheritance) {
  llvm::StringRef Test = R"cpp(
    struct $0^Base {
      virtual void F$1^oo();
      void C$4^oncrete();
    };
    struct $0[[Child1]] : Base {
      void $1[[Fo$3^o]]() override;
      virtual void B$2^ar();
      void Concrete();  // No implementations for concrete methods.
    };
    struct Child2 : Child1 {
      void $3[[Foo]]() override;
      void $2[[Bar]]() override;
    };
    void FromReference() {
      $0^Base* B;
      B->Fo$1^o();
      B->C$4^oncrete();
      &Base::Fo$1^o;
      Child1 * C1;
      C1->B$2^ar();
      C1->Fo$3^o();
    }
    // CRTP should work.
    template<typename T>
    struct $5^TemplateBase {};
    struct $5[[Child3]] : public TemplateBase<Child3> {};

    // Local classes.
    void LocationFunction() {
      struct $0[[LocalClass1]] : Base {
        void $1[[Foo]]() override;
      };
      struct $6^LocalBase {
        virtual void $7^Bar();
      };
      struct $6[[LocalClass2]]: LocalBase {
        void $7[[Bar]]() override;
      };
    }
  )cpp";

  Annotations Code(Test);
  auto TU = TestTU::withCode(Code.code());
  auto AST = TU.build();
  auto Index = TU.index();
  for (StringRef Label : {"0", "1", "2", "3", "4", "5", "6", "7"}) {
    for (const auto &Point : Code.points(Label)) {
      EXPECT_THAT(findImplementations(AST, Point, Index.get()),
                  UnorderedPointwise(declRange(), Code.ranges(Label)))
          << Code.code() << " at " << Point << " for Label " << Label;
    }
  }
}

TEST(FindImplementations, CaptureDefinition) {
  llvm::StringRef Test = R"cpp(
    struct Base {
      virtual void F^oo();
    };
    struct Child1 : Base {
      void $Decl[[Foo]]() override;
    };
    struct Child2 : Base {
      void $Child2[[Foo]]() override;
    };
    void Child1::$Def[[Foo]]() { /* Definition */ }
  )cpp";
  Annotations Code(Test);
  auto TU = TestTU::withCode(Code.code());
  auto AST = TU.build();
  EXPECT_THAT(
      findImplementations(AST, Code.point(), TU.index().get()),
      UnorderedElementsAre(sym("Foo", Code.range("Decl"), Code.range("Def")),
                           sym("Foo", Code.range("Child2"), std::nullopt)))
      << Test;
}

TEST(FindType, All) {
  Annotations HeaderA(R"cpp(
    struct $Target[[Target]] { operator int() const; };
    struct Aggregate { Target a, b; };
    Target t;
    Target make();

    template <typename T> struct $smart_ptr[[smart_ptr]] {
      T& operator*();
      T* operator->();
      T* get();
    };
  )cpp");
  auto TU = TestTU::withHeaderCode(HeaderA.code());
  for (const llvm::StringRef Case : {
           "str^uct Target;",
           "T^arget x;",
           "Target ^x;",
           "a^uto x = Target{};",
           "namespace m { Target tgt; } auto x = m^::tgt;",
           "Target funcCall(); auto x = ^funcCall();",
           "Aggregate a = { {}, ^{} };",
           "Aggregate a = { ^.a=t, };",
           "struct X { Target a; X() : ^a() {} };",
           "^using T = Target; ^T foo();",
           "^template <int> Target foo();",
           "void x() { try {} ^catch(Target e) {} }",
           "void x() { ^throw t; }",
           "int x() { ^return t; }",
           "void x() { ^switch(t) {} }",
           "void x() { ^delete (Target*)nullptr; }",
           "Target& ^tref = t;",
           "void x() { ^if (t) {} }",
           "void x() { ^while (t) {} }",
           "void x() { ^do { } while (t); }",
           "void x() { ^make(); }",
           "void x(smart_ptr<Target> &t) { t.^get(); }",
           "^auto x = []() { return t; };",
           "Target* ^tptr = &t;",
           "Target ^tarray[3];",
       }) {
    Annotations A(Case);
    TU.Code = A.code().str();
    ParsedAST AST = TU.build();

    ASSERT_GT(A.points().size(), 0u) << Case;
    for (auto Pos : A.points())
      EXPECT_THAT(findType(AST, Pos, nullptr),
                  ElementsAre(
                    sym("Target", HeaderA.range("Target"), HeaderA.range("Target"))))
          << Case;
  }

  for (const llvm::StringRef Case : {
           "smart_ptr<Target> ^tsmart;",
       }) {
    Annotations A(Case);
    TU.Code = A.code().str();
    ParsedAST AST = TU.build();

    EXPECT_THAT(findType(AST, A.point(), nullptr),
                UnorderedElementsAre(
                  sym("Target", HeaderA.range("Target"), HeaderA.range("Target")),
                  sym("smart_ptr", HeaderA.range("smart_ptr"), HeaderA.range("smart_ptr"))
                ))
        << Case;
  }
}

TEST(FindType, Definition) {
  Annotations A(R"cpp(
    class $decl[[X]];
    X *^x;
    class $def[[X]] {};
  )cpp");
  auto TU = TestTU::withCode(A.code().str());
  ParsedAST AST = TU.build();

  EXPECT_THAT(findType(AST, A.point(), nullptr),
              ElementsAre(sym("X", A.range("decl"), A.range("def"))));
}

TEST(FindType, Index) {
  Annotations Def(R"cpp(
    // This definition is only available through the index.
    class [[X]] {};
  )cpp");
  TestTU DefTU = TestTU::withHeaderCode(Def.code());
  DefTU.HeaderFilename = "def.h";
  auto DefIdx = DefTU.index();

  Annotations A(R"cpp(
    class [[X]];
    X *^x;
  )cpp");
  auto TU = TestTU::withCode(A.code().str());
  ParsedAST AST = TU.build();

  EXPECT_THAT(findType(AST, A.point(), DefIdx.get()),
              ElementsAre(sym("X", A.range(), Def.range())));
}

void checkFindRefs(llvm::StringRef Test, bool UseIndex = false) {
  Annotations T(Test);
  auto TU = TestTU::withCode(T.code());
  TU.ExtraArgs.push_back("-std=c++20");

  auto AST = TU.build();
  std::vector<Matcher<ReferencesResult::Reference>> ExpectedLocations;
  for (const auto &[R, Context] : T.rangesWithPayload())
    ExpectedLocations.push_back(
        AllOf(rangeIs(R), containerIs(Context), attrsAre(0u)));
  // $def is actually shorthand for both definition and declaration.
  // If we have cases that are definition-only, we should change this.
  for (const auto &[R, Context] : T.rangesWithPayload("def"))
    ExpectedLocations.push_back(AllOf(rangeIs(R), containerIs(Context),
                                      attrsAre(ReferencesResult::Definition |
                                               ReferencesResult::Declaration)));
  for (const auto &[R, Context] : T.rangesWithPayload("decl"))
    ExpectedLocations.push_back(AllOf(rangeIs(R), containerIs(Context),
                                      attrsAre(ReferencesResult::Declaration)));
  for (const auto &[R, Context] : T.rangesWithPayload("overridedecl"))
    ExpectedLocations.push_back(AllOf(
        rangeIs(R), containerIs(Context),
        attrsAre(ReferencesResult::Declaration | ReferencesResult::Override)));
  for (const auto &[R, Context] : T.rangesWithPayload("overridedef"))
    ExpectedLocations.push_back(AllOf(rangeIs(R), containerIs(Context),
                                      attrsAre(ReferencesResult::Declaration |
                                               ReferencesResult::Definition |
                                               ReferencesResult::Override)));
  for (const auto &P : T.points()) {
    EXPECT_THAT(findReferences(AST, P, 0, UseIndex ? TU.index().get() : nullptr,
                               /*AddContext*/ true)
                    .References,
                UnorderedElementsAreArray(ExpectedLocations))
        << "Failed for Refs at " << P << "\n"
        << Test;
  }
}

TEST(FindReferences, WithinAST) {
  const char *Tests[] = {
      R"cpp(// Local variable
        int main() {
          int $def(main)[[foo]];
          $(main)[[^foo]] = 2;
          int test1 = $(main)[[foo]];
        }
      )cpp",

      R"cpp(// Struct
        namespace ns1 {
        struct $def(ns1)[[Foo]] {};
        } // namespace ns1
        int main() {
          ns1::$(main)[[Fo^o]]* Params;
        }
      )cpp",

      R"cpp(// Forward declaration
        class $decl[[Foo]];
        class $def[[Foo]] {};
        int main() {
          $(main)[[Fo^o]] foo;
        }
      )cpp",

      R"cpp(// Function
        int $def[[foo]](int) {}
        int main() {
          auto *X = &$(main)[[^foo]];
          $(main)[[foo]](42);
        }
      )cpp",

      R"cpp(// Field
        struct Foo {
          int $def(Foo)[[foo]];
          Foo() : $(Foo::Foo)[[foo]](0) {}
        };
        int main() {
          Foo f;
          f.$(main)[[f^oo]] = 1;
        }
      )cpp",

      R"cpp(// Method call
        struct Foo { int $decl(Foo)[[foo]](); };
        int Foo::$def(Foo)[[foo]]() {}
        int main() {
          Foo f;
          f.$(main)[[^foo]]();
        }
      )cpp",

      R"cpp(// Constructor
        struct Foo {
          $decl(Foo)[[F^oo]](int);
        };
        void foo() {
          Foo f = $(foo)[[Foo]](42);
        }
      )cpp",

      R"cpp(// Typedef
        typedef int $def[[Foo]];
        int main() {
          $(main)[[^Foo]] bar;
        }
      )cpp",

      R"cpp(// Namespace
        namespace $decl[[ns]] { // FIXME: def?
        struct Foo {};
        } // namespace ns
        int main() { $(main)[[^ns]]::Foo foo; }
      )cpp",

      R"cpp(// Macros
        #define TYPE(X) X
        #define FOO Foo
        #define CAT(X, Y) X##Y
        class $def[[Fo^o]] {};
        void test() {
          TYPE($(test)[[Foo]]) foo;
          $(test)[[FOO]] foo2;
          TYPE(TYPE($(test)[[Foo]])) foo3;
          $(test)[[CAT]](Fo, o) foo4;
        }
      )cpp",

      R"cpp(// Macros
        #define $def[[MA^CRO]](X) (X+1)
        void test() {
          int x = $[[MACRO]]($[[MACRO]](1));
        }
      )cpp",

      R"cpp(// Macro outside preamble
        int breakPreamble;
        #define $def[[MA^CRO]](X) (X+1)
        void test() {
          int x = $[[MACRO]]($[[MACRO]](1));
        }
      )cpp",

      R"cpp(
        int $def[[v^ar]] = 0;
        void foo(int s = $(foo)[[var]]);
      )cpp",

      R"cpp(
       template <typename T>
       class $def[[Fo^o]] {};
       void func($(func)[[Foo]]<int>);
      )cpp",

      R"cpp(
       template <typename T>
       class $def[[Foo]] {};
       void func($(func)[[Fo^o]]<int>);
      )cpp",
      R"cpp(// Not touching any identifiers.
        struct Foo {
          $def(Foo)[[~]]Foo() {};
        };
        void foo() {
          Foo f;
          f.$(foo)[[^~]]Foo();
        }
      )cpp",
      R"cpp(// Lambda capture initializer
        void foo() {
          int $def(foo)[[w^aldo]] = 42;
          auto lambda = [x = $(foo)[[waldo]]](){};
        }
      )cpp",
      R"cpp(// Renaming alias
        template <typename> class Vector {};
        using $def[[^X]] = Vector<int>;
        $(x1)[[X]] x1;
        Vector<int> x2;
        Vector<double> y;
      )cpp",
      R"cpp(// Dependent code
        template <typename T> void $decl[[foo]](T t);
        template <typename T> void bar(T t) { $(bar)[[foo]](t); } // foo in bar is uninstantiated. 
        void baz(int x) { $(baz)[[f^oo]](x); }
      )cpp",
      R"cpp(
        namespace ns {
        struct S{};
        void $decl(ns)[[foo]](S s);
        } // namespace ns
        template <typename T> void foo(T t);
        // FIXME: Maybe report this foo as a ref to ns::foo (because of ADL)
        // when bar<ns::S> is instantiated?
        template <typename T> void bar(T t) { foo(t); }
        void baz(int x) {
          ns::S s;
          bar<ns::S>(s);
          $(baz)[[f^oo]](s);
        }
      )cpp",
      R"cpp(// unresolved member expression
        struct Foo {
          template <typename T> void $decl(Foo)[[b^ar]](T t);
        };
        template <typename T> void test(Foo F, T t) {
          F.$(test)[[bar]](t);
        }
      )cpp",

      // Enum base
      R"cpp(
        typedef int $def[[MyTypeD^ef]];
        enum MyEnum : $(MyEnum)[[MyTy^peDef]] { };
      )cpp",
      R"cpp(
        typedef int $def[[MyType^Def]];
        enum MyEnum : $(MyEnum)[[MyTypeD^ef]];
      )cpp",
      R"cpp(
        using $def[[MyTypeD^ef]] = int;
        enum MyEnum : $(MyEnum)[[MyTy^peDef]] { };
      )cpp",
  };
  for (const char *Test : Tests)
    checkFindRefs(Test);
}

TEST(FindReferences, ConceptsWithinAST) {
  constexpr llvm::StringLiteral Code = R"cpp(
    template <class T>
    concept $def[[IsSmal^l]] = sizeof(T) <= 8;

    template <class T>
    concept IsSmallPtr = requires(T x) {
      { *x } -> $(IsSmallPtr)[[IsSmal^l]];
    };

    $(i)[[IsSmall]] auto i = 'c';
    template<$(foo)[[IsSmal^l]] U> void foo();
    template<class U> void bar() requires $(bar)[[IsSmal^l]]<U>;
    template<class U> requires $(baz)[[IsSmal^l]]<U> void baz();
    static_assert([[IsSma^ll]]<char>);
  )cpp";
  checkFindRefs(Code);
}

TEST(FindReferences, ConceptReq) {
  constexpr llvm::StringLiteral Code = R"cpp(
    template <class T>
    concept $def[[IsSmal^l]] = sizeof(T) <= 8;

    template <class T>
    concept IsSmallPtr = requires(T x) {
      { *x } -> $(IsSmallPtr)[[IsSmal^l]];
    };
  )cpp";
  checkFindRefs(Code);
}

TEST(FindReferences, RequiresExprParameters) {
  constexpr llvm::StringLiteral Code = R"cpp(
    template <class T>
    concept IsSmall = sizeof(T) <= 8;

    template <class T>
    concept IsSmallPtr = requires(T $def[[^x]]) {
      { *$(IsSmallPtr)[[^x]] } -> IsSmall;
    };
  )cpp";
  checkFindRefs(Code);
}

TEST(FindReferences, IncludeOverrides) {
  llvm::StringRef Test =
      R"cpp(
        class Base {
        public:
          virtu^al void $decl(Base)[[f^unc]]() ^= ^0;
        };
        class Derived : public Base {
        public:
          void $overridedecl(Derived::func)[[func]]() override;
        };
        void Derived::$overridedef[[func]]() {}
        class Derived2 : public Base {
          void $overridedef(Derived2::func)[[func]]() override {}
        };
        void test(Derived* D, Base* B) {
          D->func();  // No references to the overrides.
          B->$(test)[[func]]();
        })cpp";
  checkFindRefs(Test, /*UseIndex=*/true);
}

TEST(FindReferences, RefsToBaseMethod) {
  llvm::StringRef Test =
      R"cpp(
        class BaseBase {
        public:
          virtual void $(BaseBase)[[func]]();
        };
        class Base : public BaseBase {
        public:
          void $(Base)[[func]]() override;
        };
        class Derived : public Base {
        public:
          void $decl(Derived)[[fu^nc]]() over^ride;
        };
        void test(BaseBase* BB, Base* B, Derived* D) {
          // refs to overridden methods in complete type hierarchy are reported.
          BB->$(test)[[func]]();
          B->$(test)[[func]]();
          D->$(test)[[fu^nc]]();
        })cpp";
  checkFindRefs(Test, /*UseIndex=*/true);
}

TEST(FindReferences, MainFileReferencesOnly) {
  llvm::StringRef Test =
      R"cpp(
        void test() {
          int [[fo^o]] = 1;
          // refs not from main file should not be included.
          #include "foo.inc"
        })cpp";

  Annotations Code(Test);
  auto TU = TestTU::withCode(Code.code());
  TU.AdditionalFiles["foo.inc"] = R"cpp(
      foo = 3;
    )cpp";
  auto AST = TU.build();

  std::vector<Matcher<ReferencesResult::Reference>> ExpectedLocations;
  for (const auto &R : Code.ranges())
    ExpectedLocations.push_back(rangeIs(R));
  EXPECT_THAT(findReferences(AST, Code.point(), 0).References,
              ElementsAreArray(ExpectedLocations))
      << Test;
}

TEST(FindReferences, ExplicitSymbols) {
  const char *Tests[] = {
      R"cpp(
      struct Foo { Foo* $decl(Foo)[[self]]() const; };
      void f() {
        Foo foo;
        if (Foo* T = foo.$(f)[[^self]]()) {} // Foo member call expr.
      }
      )cpp",

      R"cpp(
      struct Foo { Foo(int); };
      Foo f() {
        int $def(f)[[b]];
        return $(f)[[^b]]; // Foo constructor expr.
      }
      )cpp",

      R"cpp(
      struct Foo {};
      void g(Foo);
      Foo $decl[[f]]();
      void call() {
        g($(call)[[^f]]());  // Foo constructor expr.
      }
      )cpp",

      R"cpp(
      void $decl[[foo]](int);
      void $decl[[foo]](double);

      namespace ns {
      using ::$decl(ns)[[fo^o]];
      }
      )cpp",

      R"cpp(
      struct X {
        operator bool();
      };

      int test() {
        X $def(test)[[a]];
        $(test)[[a]].operator bool();
        if ($(test)[[a^]]) {} // ignore implicit conversion-operator AST node
      }
    )cpp",
  };
  for (const char *Test : Tests)
    checkFindRefs(Test);
}

TEST(FindReferences, UsedSymbolsFromInclude) {
  const char *Tests[] = {
      R"cpp(   [[#include   ^"bar.h"]]
        #include <vector>
        int fstBar = [[bar1]]();
        int sndBar = [[bar2]]();
        [[Bar]] bar;
        int macroBar = [[BAR]];
        std::vector<int> vec;
      )cpp",

      R"cpp([[#in^clude <vector>]]
        std::[[vector]]<int> vec;
      )cpp"};
  for (const char *Test : Tests) {
    Annotations T(Test);
    auto TU = TestTU::withCode(T.code());
    TU.ExtraArgs.push_back("-std=c++20");
    TU.AdditionalFiles["bar.h"] = guard(R"cpp(
      #define BAR 5
      int bar1();
      int bar2();
      class Bar {};            
    )cpp");
    TU.AdditionalFiles["system/vector"] = guard(R"cpp(
      namespace std {
        template<typename>
        class vector{};
      }
    )cpp");
    TU.ExtraArgs.push_back("-isystem" + testPath("system"));

    auto AST = TU.build();
    std::vector<Matcher<ReferencesResult::Reference>> ExpectedLocations;
    for (const auto &R : T.ranges())
      ExpectedLocations.push_back(AllOf(rangeIs(R), attrsAre(0u)));
    for (const auto &P : T.points()) 
      EXPECT_THAT(findReferences(AST, P, 0).References,
                  UnorderedElementsAreArray(ExpectedLocations))
          << "Failed for Refs at " << P << "\n"
          << Test;
  }
}

TEST(FindReferences, NeedsIndexForSymbols) {
  const char *Header = "int foo();";
  Annotations Main("int main() { [[f^oo]](); }");
  TestTU TU;
  TU.Code = std::string(Main.code());
  TU.HeaderCode = Header;
  auto AST = TU.build();

  // References in main file are returned without index.
  EXPECT_THAT(
      findReferences(AST, Main.point(), 0, /*Index=*/nullptr).References,
      ElementsAre(rangeIs(Main.range())));
  Annotations IndexedMain(R"cpp(
    int $decl[[foo]]() { return 42; }
    void bar() { $bar(bar)[[foo]](); }
    struct S { void bar() { $S(S::bar)[[foo]](); } };
    namespace N { void bar() { $N(N::bar)[[foo]](); } }
  )cpp");

  // References from indexed files are included.
  TestTU IndexedTU;
  IndexedTU.Code = std::string(IndexedMain.code());
  IndexedTU.Filename = "Indexed.cpp";
  IndexedTU.HeaderCode = Header;
  EXPECT_THAT(
      findReferences(AST, Main.point(), 0, IndexedTU.index().get(),
                     /*AddContext*/ true)
          .References,
      ElementsAre(
          rangeIs(Main.range()),
          AllOf(rangeIs(IndexedMain.range("decl")),
                attrsAre(ReferencesResult::Declaration |
                         ReferencesResult::Definition)),
          AllOf(rangeIs(IndexedMain.range("bar")), containerIs("bar")),
          AllOf(rangeIs(IndexedMain.range("S")), containerIs("S::bar")),
          AllOf(rangeIs(IndexedMain.range("N")), containerIs("N::bar"))));
  auto LimitRefs =
      findReferences(AST, Main.point(), /*Limit*/ 1, IndexedTU.index().get());
  EXPECT_EQ(1u, LimitRefs.References.size());
  EXPECT_TRUE(LimitRefs.HasMore);

  // Avoid indexed results for the main file. Use AST for the mainfile.
  TU.Code = ("\n\n" + Main.code()).str();
  EXPECT_THAT(findReferences(AST, Main.point(), 0, TU.index().get()).References,
              ElementsAre(rangeIs(Main.range())));
}

TEST(FindReferences, NeedsIndexForMacro) {
  const char *Header = "#define MACRO(X) (X+1)";
  Annotations Main(R"cpp(
    int main() {
      int a = [[MA^CRO]](1);
    }
  )cpp");
  TestTU TU;
  TU.Code = std::string(Main.code());
  TU.HeaderCode = Header;
  auto AST = TU.build();

  // References in main file are returned without index.
  EXPECT_THAT(
      findReferences(AST, Main.point(), 0, /*Index=*/nullptr).References,
      ElementsAre(rangeIs(Main.range())));

  Annotations IndexedMain(R"cpp(
    int indexed_main() {
      int a = [[MACRO]](1);
    }
  )cpp");

  // References from indexed files are included.
  TestTU IndexedTU;
  IndexedTU.Code = std::string(IndexedMain.code());
  IndexedTU.Filename = "Indexed.cpp";
  IndexedTU.HeaderCode = Header;
  EXPECT_THAT(
      findReferences(AST, Main.point(), 0, IndexedTU.index().get()).References,
      ElementsAre(rangeIs(Main.range()), rangeIs(IndexedMain.range())));
  auto LimitRefs =
      findReferences(AST, Main.point(), /*Limit*/ 1, IndexedTU.index().get());
  EXPECT_EQ(1u, LimitRefs.References.size());
  EXPECT_TRUE(LimitRefs.HasMore);
}

TEST(FindReferences, NoQueryForLocalSymbols) {
  struct RecordingIndex : public MemIndex {
    mutable std::optional<llvm::DenseSet<SymbolID>> RefIDs;
    bool refs(const RefsRequest &Req,
              llvm::function_ref<void(const Ref &)>) const override {
      RefIDs = Req.IDs;
      return false;
    }
  };

  struct Test {
    StringRef AnnotatedCode;
    bool WantQuery;
  } Tests[] = {
      {"int ^x;", true},
      // For now we don't assume header structure which would allow skipping.
      {"namespace { int ^x; }", true},
      {"static int ^x;", true},
      // Anything in a function certainly can't be referenced though.
      {"void foo() { int ^x; }", false},
      {"void foo() { struct ^x{}; }", false},
      {"auto lambda = []{ int ^x; };", false},
  };
  for (Test T : Tests) {
    Annotations File(T.AnnotatedCode);
    RecordingIndex Rec;
    auto AST = TestTU::withCode(File.code()).build();
    findReferences(AST, File.point(), 0, &Rec);
    if (T.WantQuery)
      EXPECT_NE(Rec.RefIDs, std::nullopt) << T.AnnotatedCode;
    else
      EXPECT_EQ(Rec.RefIDs, std::nullopt) << T.AnnotatedCode;
  }
}

TEST(GetNonLocalDeclRefs, All) {
  struct Case {
    llvm::StringRef AnnotatedCode;
    std::vector<std::string> ExpectedDecls;
  } Cases[] = {
      {
          // VarDecl and ParamVarDecl
          R"cpp(
            void bar();
            void ^foo(int baz) {
              int x = 10;
              bar();
            })cpp",
          {"bar"},
      },
      {
          // Method from class
          R"cpp(
            class Foo { public: void foo(); };
            class Bar {
              void foo();
              void bar();
            };
            void Bar::^foo() {
              Foo f;
              bar();
              f.foo();
            })cpp",
          {"Bar", "Bar::bar", "Foo", "Foo::foo"},
      },
      {
          // Local types
          R"cpp(
            void ^foo() {
              class Foo { public: void foo() {} };
              class Bar { public: void bar() {} };
              Foo f;
              Bar b;
              b.bar();
              f.foo();
            })cpp",
          {},
      },
      {
          // Template params
          R"cpp(
            template <typename T, template<typename> class Q>
            void ^foo() {
              T x;
              Q<T> y;
            })cpp",
          {},
      },
  };
  for (const Case &C : Cases) {
    Annotations File(C.AnnotatedCode);
    auto AST = TestTU::withCode(File.code()).build();
    SourceLocation SL = llvm::cantFail(
        sourceLocationInMainFile(AST.getSourceManager(), File.point()));

    const FunctionDecl *FD =
        llvm::dyn_cast<FunctionDecl>(&findDecl(AST, [SL](const NamedDecl &ND) {
          return ND.getLocation() == SL && llvm::isa<FunctionDecl>(ND);
        }));
    ASSERT_NE(FD, nullptr);

    auto NonLocalDeclRefs = getNonLocalDeclRefs(AST, FD);
    std::vector<std::string> Names;
    for (const Decl *D : NonLocalDeclRefs) {
      if (const auto *ND = llvm::dyn_cast<NamedDecl>(D))
        Names.push_back(ND->getQualifiedNameAsString());
    }
    EXPECT_THAT(Names, UnorderedElementsAreArray(C.ExpectedDecls))
        << File.code();
  }
}

TEST(DocumentLinks, All) {
  Annotations MainCpp(R"cpp(
      #/*comments*/include /*comments*/ $foo[["foo.h"]] //more comments
      int end_of_preamble = 0;
      #include $bar[[<bar.h>]]
    )cpp");

  TestTU TU;
  TU.Code = std::string(MainCpp.code());
  TU.AdditionalFiles = {{"foo.h", ""}, {"bar.h", ""}};
  TU.ExtraArgs = {"-isystem."};
  auto AST = TU.build();

  EXPECT_THAT(
      clangd::getDocumentLinks(AST),
      ElementsAre(
          DocumentLink({MainCpp.range("foo"),
                        URIForFile::canonicalize(testPath("foo.h"), "")}),
          DocumentLink({MainCpp.range("bar"),
                        URIForFile::canonicalize(testPath("bar.h"), "")})));
}

} // namespace
} // namespace clangd
} // namespace clang
