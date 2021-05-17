//===-- FindSymbolsTests.cpp -------------------------*- C++ -*------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Annotations.h"
#include "ClangdServer.h"
#include "FindSymbols.h"
#include "SyncAPI.h"
#include "TestFS.h"
#include "TestTU.h"
#include "llvm/ADT/StringRef.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {

namespace {

using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Field;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

// GMock helpers for matching SymbolInfos items.
MATCHER_P(QName, Name, "") {
  if (arg.containerName.empty())
    return arg.name == Name;
  return (arg.containerName + "::" + arg.name) == Name;
}
MATCHER_P(WithName, N, "") { return arg.name == N; }
MATCHER_P(WithKind, Kind, "") { return arg.kind == Kind; }
MATCHER_P(WithDetail, Detail, "") { return arg.detail == Detail; }
MATCHER_P(SymRange, Range, "") { return arg.range == Range; }

// GMock helpers for matching DocumentSymbol.
MATCHER_P(SymNameRange, Range, "") { return arg.selectionRange == Range; }
template <class... ChildMatchers>
::testing::Matcher<DocumentSymbol> Children(ChildMatchers... ChildrenM) {
  return Field(&DocumentSymbol::children, ElementsAre(ChildrenM...));
}

std::vector<SymbolInformation> getSymbols(TestTU &TU, llvm::StringRef Query,
                                          int Limit = 0) {
  auto SymbolInfos = getWorkspaceSymbols(Query, Limit, TU.index().get(),
                                         testPath(TU.Filename));
  EXPECT_TRUE(bool(SymbolInfos)) << "workspaceSymbols returned an error";
  return *SymbolInfos;
}

TEST(WorkspaceSymbols, Macros) {
  TestTU TU;
  TU.Code = R"cpp(
       #define MACRO X
       )cpp";

  // LSP's SymbolKind doesn't have a "Macro" kind, and
  // indexSymbolKindToSymbolKind() currently maps macros
  // to SymbolKind::String.
  EXPECT_THAT(getSymbols(TU, "macro"),
              ElementsAre(AllOf(QName("MACRO"), WithKind(SymbolKind::String))));
}

TEST(WorkspaceSymbols, NoLocals) {
  TestTU TU;
  TU.Code = R"cpp(
      void test(int FirstParam, int SecondParam) {
        struct LocalClass {};
        int local_var;
      })cpp";
  EXPECT_THAT(getSymbols(TU, "l"), ElementsAre(QName("LocalClass")));
  EXPECT_THAT(getSymbols(TU, "p"), IsEmpty());
}

TEST(WorkspaceSymbols, Globals) {
  TestTU TU;
  TU.AdditionalFiles["foo.h"] = R"cpp(
      int global_var;

      int global_func();

      struct GlobalStruct {};)cpp";
  TU.Code = R"cpp(
      #include "foo.h"
      )cpp";
  EXPECT_THAT(getSymbols(TU, "global"),
              UnorderedElementsAre(
                  AllOf(QName("GlobalStruct"), WithKind(SymbolKind::Struct)),
                  AllOf(QName("global_func"), WithKind(SymbolKind::Function)),
                  AllOf(QName("global_var"), WithKind(SymbolKind::Variable))));
}

TEST(WorkspaceSymbols, Unnamed) {
  TestTU TU;
  TU.AdditionalFiles["foo.h"] = R"cpp(
      struct {
        int InUnnamed;
      } UnnamedStruct;)cpp";
  TU.Code = R"cpp(
      #include "foo.h"
      )cpp";
  EXPECT_THAT(getSymbols(TU, "UnnamedStruct"),
              ElementsAre(AllOf(QName("UnnamedStruct"),
                                WithKind(SymbolKind::Variable))));
  EXPECT_THAT(getSymbols(TU, "InUnnamed"),
              ElementsAre(AllOf(QName("(anonymous struct)::InUnnamed"),
                                WithKind(SymbolKind::Field))));
}

TEST(WorkspaceSymbols, InMainFile) {
  TestTU TU;
  TU.Code = R"cpp(
      int test() {}
      static void test2() {}
      )cpp";
  EXPECT_THAT(getSymbols(TU, "test"),
              ElementsAre(QName("test"), QName("test2")));
}

TEST(WorkspaceSymbols, Namespaces) {
  TestTU TU;
  TU.AdditionalFiles["foo.h"] = R"cpp(
      namespace ans1 {
        int ai1;
        namespace ans2 {
          int ai2;
          namespace ans3 {
            int ai3;
          }
        }
      }
      )cpp";
  TU.Code = R"cpp(
      #include "foo.h"
      )cpp";
  EXPECT_THAT(getSymbols(TU, "a"),
              UnorderedElementsAre(
                  QName("ans1"), QName("ans1::ai1"), QName("ans1::ans2"),
                  QName("ans1::ans2::ai2"), QName("ans1::ans2::ans3"),
                  QName("ans1::ans2::ans3::ai3")));
  EXPECT_THAT(getSymbols(TU, "::"), ElementsAre(QName("ans1")));
  EXPECT_THAT(getSymbols(TU, "::a"), ElementsAre(QName("ans1")));
  EXPECT_THAT(getSymbols(TU, "ans1::"),
              UnorderedElementsAre(QName("ans1::ai1"), QName("ans1::ans2"),
                                   QName("ans1::ans2::ai2"),
                                   QName("ans1::ans2::ans3"),
                                   QName("ans1::ans2::ans3::ai3")));
  EXPECT_THAT(getSymbols(TU, "ans2::"),
              UnorderedElementsAre(QName("ans1::ans2::ai2"),
                                   QName("ans1::ans2::ans3"),
                                   QName("ans1::ans2::ans3::ai3")));
  EXPECT_THAT(getSymbols(TU, "::ans1"), ElementsAre(QName("ans1")));
  EXPECT_THAT(getSymbols(TU, "::ans1::"),
              UnorderedElementsAre(QName("ans1::ai1"), QName("ans1::ans2")));
  EXPECT_THAT(getSymbols(TU, "::ans1::ans2"), ElementsAre(QName("ans1::ans2")));
  EXPECT_THAT(getSymbols(TU, "::ans1::ans2::"),
              ElementsAre(QName("ans1::ans2::ai2"), QName("ans1::ans2::ans3")));

  // Ensure sub-sequence matching works.
  EXPECT_THAT(getSymbols(TU, "ans1::ans3::ai"),
              UnorderedElementsAre(QName("ans1::ans2::ans3::ai3")));
}

TEST(WorkspaceSymbols, AnonymousNamespace) {
  TestTU TU;
  TU.Code = R"cpp(
      namespace {
      void test() {}
      }
      )cpp";
  EXPECT_THAT(getSymbols(TU, "test"), ElementsAre(QName("test")));
}

TEST(WorkspaceSymbols, MultiFile) {
  TestTU TU;
  TU.AdditionalFiles["foo.h"] = R"cpp(
      int foo() {
      }
      )cpp";
  TU.AdditionalFiles["foo2.h"] = R"cpp(
      int foo2() {
      }
      )cpp";
  TU.Code = R"cpp(
      #include "foo.h"
      #include "foo2.h"
      )cpp";
  EXPECT_THAT(getSymbols(TU, "foo"),
              UnorderedElementsAre(QName("foo"), QName("foo2")));
}

TEST(WorkspaceSymbols, GlobalNamespaceQueries) {
  TestTU TU;
  TU.AdditionalFiles["foo.h"] = R"cpp(
      int foo() {
      }
      class Foo {
        int a;
      };
      namespace ns {
      int foo2() {
      }
      }
      )cpp";
  TU.Code = R"cpp(
      #include "foo.h"
      )cpp";
  EXPECT_THAT(getSymbols(TU, "::"),
              UnorderedElementsAre(
                  AllOf(QName("Foo"), WithKind(SymbolKind::Class)),
                  AllOf(QName("foo"), WithKind(SymbolKind::Function)),
                  AllOf(QName("ns"), WithKind(SymbolKind::Namespace))));
  EXPECT_THAT(getSymbols(TU, ":"), IsEmpty());
  EXPECT_THAT(getSymbols(TU, ""),
              UnorderedElementsAre(QName("foo"), QName("Foo"), QName("Foo::a"),
                                   QName("ns"), QName("ns::foo2")));
}

TEST(WorkspaceSymbols, Enums) {
  TestTU TU;
  TU.AdditionalFiles["foo.h"] = R"cpp(
    enum {
      Red
    };
    enum Color {
      Green
    };
    enum class Color2 {
      Yellow
    };
    namespace ns {
      enum {
        Black
      };
      enum Color3 {
        Blue
      };
      enum class Color4 {
        White
      };
    }
      )cpp";
  TU.Code = R"cpp(
      #include "foo.h"
      )cpp";
  EXPECT_THAT(getSymbols(TU, "Red"), ElementsAre(QName("Red")));
  EXPECT_THAT(getSymbols(TU, "::Red"), ElementsAre(QName("Red")));
  EXPECT_THAT(getSymbols(TU, "Green"), ElementsAre(QName("Green")));
  EXPECT_THAT(getSymbols(TU, "Green"), ElementsAre(QName("Green")));
  EXPECT_THAT(getSymbols(TU, "Color2::Yellow"),
              ElementsAre(QName("Color2::Yellow")));
  EXPECT_THAT(getSymbols(TU, "Yellow"), ElementsAre(QName("Color2::Yellow")));

  EXPECT_THAT(getSymbols(TU, "ns::Black"), ElementsAre(QName("ns::Black")));
  EXPECT_THAT(getSymbols(TU, "ns::Blue"), ElementsAre(QName("ns::Blue")));
  EXPECT_THAT(getSymbols(TU, "ns::Color4::White"),
              ElementsAre(QName("ns::Color4::White")));
}

TEST(WorkspaceSymbols, Ranking) {
  TestTU TU;
  TU.AdditionalFiles["foo.h"] = R"cpp(
      namespace ns{}
      void func();
      )cpp";
  TU.Code = R"cpp(
      #include "foo.h"
      )cpp";
  EXPECT_THAT(getSymbols(TU, "::"), ElementsAre(QName("func"), QName("ns")));
}

TEST(WorkspaceSymbols, RankingPartialNamespace) {
  TestTU TU;
  TU.Code = R"cpp(
    namespace ns1 {
      namespace ns2 { struct Foo {}; }
    }
    namespace ns2 { struct FooB {}; })cpp";
  EXPECT_THAT(getSymbols(TU, "ns2::f"),
              ElementsAre(QName("ns2::FooB"), QName("ns1::ns2::Foo")));
}

TEST(WorkspaceSymbols, WithLimit) {
  TestTU TU;
  TU.AdditionalFiles["foo.h"] = R"cpp(
      int foo;
      int foo2;
      )cpp";
  TU.Code = R"cpp(
      #include "foo.h"
      )cpp";
  // Foo is higher ranked because of exact name match.
  EXPECT_THAT(getSymbols(TU, "foo"),
              UnorderedElementsAre(
                  AllOf(QName("foo"), WithKind(SymbolKind::Variable)),
                  AllOf(QName("foo2"), WithKind(SymbolKind::Variable))));

  EXPECT_THAT(getSymbols(TU, "foo", 1), ElementsAre(QName("foo")));
}

TEST(WorkspaceSymbols, TempSpecs) {
  TestTU TU;
  TU.ExtraArgs = {"-xc++"};
  TU.Code = R"cpp(
      template <typename T, typename U, int X = 5> class Foo {};
      template <typename T> class Foo<int, T> {};
      template <> class Foo<bool, int> {};
      template <> class Foo<bool, int, 3> {};
      )cpp";
  // Foo is higher ranked because of exact name match.
  EXPECT_THAT(
      getSymbols(TU, "Foo"),
      UnorderedElementsAre(
          AllOf(QName("Foo"), WithKind(SymbolKind::Class)),
          AllOf(QName("Foo<int, T>"), WithKind(SymbolKind::Class)),
          AllOf(QName("Foo<bool, int>"), WithKind(SymbolKind::Class)),
          AllOf(QName("Foo<bool, int, 3>"), WithKind(SymbolKind::Class))));
}

std::vector<DocumentSymbol> getSymbols(ParsedAST AST) {
  auto SymbolInfos = getDocumentSymbols(AST);
  EXPECT_TRUE(bool(SymbolInfos)) << "documentSymbols returned an error";
  return *SymbolInfos;
}

TEST(DocumentSymbols, BasicSymbols) {
  TestTU TU;
  Annotations Main(R"(
      class Foo;
      class Foo {
        Foo() {}
        Foo(int a) {}
        void $decl[[f]]();
        friend void f1();
        friend class Friend;
        Foo& operator=(const Foo&);
        ~Foo();
        class Nested {
        void f();
        };
      };
      class Friend {
      };

      void f1();
      inline void f2() {}
      static const int KInt = 2;
      const char* kStr = "123";

      void f1() {}

      namespace foo {
      // Type alias
      typedef int int32;
      using int32_t = int32;

      // Variable
      int v1;

      // Namespace
      namespace bar {
      int v2;
      }
      // Namespace alias
      namespace baz = bar;

      using bar::v2;
      } // namespace foo
    )");

  TU.Code = Main.code().str();
  EXPECT_THAT(
      getSymbols(TU.build()),
      ElementsAreArray(
          {AllOf(WithName("Foo"), WithKind(SymbolKind::Class),
                 WithDetail("class"), Children()),
           AllOf(WithName("Foo"), WithKind(SymbolKind::Class),
                 WithDetail("class"),
                 Children(
                     AllOf(WithName("Foo"), WithKind(SymbolKind::Constructor),
                           WithDetail("()"), Children()),
                     AllOf(WithName("Foo"), WithKind(SymbolKind::Constructor),
                           WithDetail("(int)"), Children()),
                     AllOf(WithName("f"), WithKind(SymbolKind::Method),
                           WithDetail("void ()"), Children()),
                     AllOf(WithName("operator="), WithKind(SymbolKind::Method),
                           WithDetail("Foo &(const Foo &)"), Children()),
                     AllOf(WithName("~Foo"), WithKind(SymbolKind::Constructor),
                           WithDetail(""), Children()),
                     AllOf(WithName("Nested"), WithKind(SymbolKind::Class),
                           WithDetail("class"),
                           Children(AllOf(
                               WithName("f"), WithKind(SymbolKind::Method),
                               WithDetail("void ()"), Children()))))),
           AllOf(WithName("Friend"), WithKind(SymbolKind::Class),
                 WithDetail("class"), Children()),
           AllOf(WithName("f1"), WithKind(SymbolKind::Function),
                 WithDetail("void ()"), Children()),
           AllOf(WithName("f2"), WithKind(SymbolKind::Function),
                 WithDetail("void ()"), Children()),
           AllOf(WithName("KInt"), WithKind(SymbolKind::Variable),
                 WithDetail("const int"), Children()),
           AllOf(WithName("kStr"), WithKind(SymbolKind::Variable),
                 WithDetail("const char *"), Children()),
           AllOf(WithName("f1"), WithKind(SymbolKind::Function),
                 WithDetail("void ()"), Children()),
           AllOf(
               WithName("foo"), WithKind(SymbolKind::Namespace), WithDetail(""),
               Children(AllOf(WithName("int32"), WithKind(SymbolKind::Class),
                              WithDetail("type alias"), Children()),
                        AllOf(WithName("int32_t"), WithKind(SymbolKind::Class),
                              WithDetail("type alias"), Children()),
                        AllOf(WithName("v1"), WithKind(SymbolKind::Variable),
                              WithDetail("int"), Children()),
                        AllOf(WithName("bar"), WithKind(SymbolKind::Namespace),
                              WithDetail(""),
                              Children(AllOf(WithName("v2"),
                                             WithKind(SymbolKind::Variable),
                                             WithDetail("int"), Children()))),
                        AllOf(WithName("baz"), WithKind(SymbolKind::Namespace),
                              WithDetail(""), Children()),
                        AllOf(WithName("v2"), WithKind(SymbolKind::Namespace),
                              WithDetail(""))))}));
}

TEST(DocumentSymbols, DeclarationDefinition) {
  TestTU TU;
  Annotations Main(R"(
      class Foo {
        void $decl[[f]]();
      };
      void Foo::$def[[f]]() {
      }
    )");

  TU.Code = Main.code().str();
  EXPECT_THAT(
      getSymbols(TU.build()),
      ElementsAre(
          AllOf(WithName("Foo"), WithKind(SymbolKind::Class),
                WithDetail("class"),
                Children(AllOf(WithName("f"), WithKind(SymbolKind::Method),
                               WithDetail("void ()"),
                               SymNameRange(Main.range("decl"))))),
          AllOf(WithName("Foo::f"), WithKind(SymbolKind::Method),
                WithDetail("void ()"), SymNameRange(Main.range("def")))));
}

TEST(DocumentSymbols, Concepts) {
  TestTU TU;
  TU.ExtraArgs = {"-std=c++20"};
  TU.Code = "template <typename T> concept C = requires(T t) { t.foo(); };";

  EXPECT_THAT(getSymbols(TU.build()),
              ElementsAre(AllOf(WithName("C"), WithDetail("concept"))));
}

TEST(DocumentSymbols, ExternSymbol) {
  TestTU TU;
  TU.AdditionalFiles["foo.h"] = R"cpp(
      extern int var;
      )cpp";
  TU.Code = R"cpp(
      #include "foo.h"
      )cpp";

  EXPECT_THAT(getSymbols(TU.build()), IsEmpty());
}

TEST(DocumentSymbols, ExternContext) {
  TestTU TU;
  TU.Code = R"cpp(
      extern "C" {
      void foo();
      class Foo {};
      }
      namespace ns {
        extern "C" {
        void bar();
        class Bar {};
        }
      })cpp";

  EXPECT_THAT(getSymbols(TU.build()),
              ElementsAre(WithName("foo"), WithName("Foo"),
                          AllOf(WithName("ns"),
                                Children(WithName("bar"), WithName("Bar")))));
}

TEST(DocumentSymbols, ExportContext) {
  TestTU TU;
  TU.ExtraArgs = {"-std=c++20"};
  TU.Code = R"cpp(
      export module test;
      export {
      void foo();
      class Foo {};
      })cpp";

  EXPECT_THAT(getSymbols(TU.build()),
              ElementsAre(WithName("foo"), WithName("Foo")));
}

TEST(DocumentSymbols, NoLocals) {
  TestTU TU;
  TU.Code = R"cpp(
      void test(int FirstParam, int SecondParam) {
        struct LocalClass {};
        int local_var;
      })cpp";
  EXPECT_THAT(getSymbols(TU.build()), ElementsAre(WithName("test")));
}

TEST(DocumentSymbols, Unnamed) {
  TestTU TU;
  TU.Code = R"cpp(
      struct {
        int InUnnamed;
      } UnnamedStruct;
      )cpp";
  EXPECT_THAT(
      getSymbols(TU.build()),
      ElementsAre(AllOf(WithName("(anonymous struct)"),
                        WithKind(SymbolKind::Struct), WithDetail("struct"),
                        Children(AllOf(WithName("InUnnamed"),
                                       WithKind(SymbolKind::Field),
                                       WithDetail("int"), Children()))),
                  AllOf(WithName("UnnamedStruct"),
                        WithKind(SymbolKind::Variable),
                        WithDetail("struct (unnamed)"), Children())));
}

TEST(DocumentSymbols, InHeaderFile) {
  TestTU TU;
  TU.AdditionalFiles["bar.h"] = R"cpp(
      int foo() {
      }
      )cpp";
  TU.Code = R"cpp(
      int i; // declaration to finish preamble
      #include "bar.h"
      int test() {
      }
      )cpp";
  EXPECT_THAT(getSymbols(TU.build()),
              ElementsAre(WithName("i"), WithName("test")));
}

TEST(DocumentSymbols, Template) {
  TestTU TU;
  TU.Code = R"(
    template <class T> struct Tmpl {T x = 0;};
    template <> struct Tmpl<int> {
      int y = 0;
    };
    extern template struct Tmpl<float>;
    template struct Tmpl<double>;

    template <class T, class U, class Z = float>
    int funcTmpl(U a);
    template <>
    int funcTmpl<int>(double a);

    template <class T, class U = double>
    int varTmpl = T();
    template <>
    double varTmpl<int> = 10.0;
  )";
  EXPECT_THAT(
      getSymbols(TU.build()),
      ElementsAre(
          AllOf(WithName("Tmpl"), WithKind(SymbolKind::Struct),
                WithDetail("template struct"),
                Children(AllOf(WithName("x"), WithKind(SymbolKind::Field),
                               WithDetail("T")))),
          AllOf(WithName("Tmpl<int>"), WithKind(SymbolKind::Struct),
                WithDetail("struct"),
                Children(AllOf(WithName("y"), WithDetail("int")))),
          AllOf(WithName("Tmpl<float>"), WithKind(SymbolKind::Struct),
                WithDetail("struct"), Children()),
          AllOf(WithName("Tmpl<double>"), WithKind(SymbolKind::Struct),
                WithDetail("struct"), Children()),
          AllOf(WithName("funcTmpl"), WithDetail("template int (U)"),
                Children()),
          AllOf(WithName("funcTmpl<int>"), WithDetail("int (double)"),
                Children()),
          AllOf(WithName("varTmpl"), WithDetail("template int"), Children()),
          AllOf(WithName("varTmpl<int>"), WithDetail("double"), Children())));
}

TEST(DocumentSymbols, Namespaces) {
  TestTU TU;
  TU.Code = R"cpp(
      namespace ans1 {
        int ai1;
      namespace ans2 {
        int ai2;
      }
      }
      namespace {
      void test() {}
      }

      namespace na {
      inline namespace nb {
      class Foo {};
      }
      }
      namespace na {
      // This is still inlined.
      namespace nb {
      class Bar {};
      }
      }
      )cpp";
  EXPECT_THAT(
      getSymbols(TU.build()),
      ElementsAreArray<::testing::Matcher<DocumentSymbol>>(
          {AllOf(WithName("ans1"),
                 Children(AllOf(WithName("ai1"), Children()),
                          AllOf(WithName("ans2"), Children(WithName("ai2"))))),
           AllOf(WithName("(anonymous namespace)"), Children(WithName("test"))),
           AllOf(WithName("na"),
                 Children(AllOf(WithName("nb"), Children(WithName("Foo"))))),
           AllOf(WithName("na"),
                 Children(AllOf(WithName("nb"), Children(WithName("Bar")))))}));
}

TEST(DocumentSymbols, Enums) {
  TestTU TU;
  TU.Code = R"(
      enum {
        Red
      };
      enum Color {
        Green
      };
      enum class Color2 {
        Yellow
      };
      namespace ns {
      enum {
        Black
      };
      }
    )";
  EXPECT_THAT(
      getSymbols(TU.build()),
      ElementsAre(
          AllOf(WithName("(anonymous enum)"), WithDetail("enum"), 
                Children(AllOf(WithName("Red"), WithDetail("(unnamed)")))),
          AllOf(WithName("Color"), WithDetail("enum"),
                Children(AllOf(WithName("Green"), WithDetail("Color")))),
          AllOf(WithName("Color2"), WithDetail("enum"),
                Children(AllOf(WithName("Yellow"), WithDetail("Color2")))),
          AllOf(WithName("ns"),
                Children(AllOf(WithName("(anonymous enum)"), WithDetail("enum"),
                               Children(AllOf(WithName("Black"),
                                              WithDetail("(unnamed)"))))))));
}

TEST(DocumentSymbols, Macro) {
  struct Test {
    const char *Code;
    testing::Matcher<DocumentSymbol> Matcher;
  } Tests[] = {
      {
          R"cpp(
            // Basic macro that generates symbols.
            #define DEFINE_FLAG(X) bool FLAGS_##X; bool FLAGS_no##X
            DEFINE_FLAG(pretty);
          )cpp",
          AllOf(WithName("DEFINE_FLAG"), WithDetail("(pretty)"),
                Children(WithName("FLAGS_pretty"), WithName("FLAGS_nopretty"))),
      },
      {
          R"cpp(
            // Hierarchy is determined by primary (name) location.
            #define ID(X) X
            namespace ID(ns) { int ID(y); }
          )cpp",
          AllOf(WithName("ID"), WithDetail("(ns)"),
                Children(AllOf(WithName("ns"),
                               Children(AllOf(WithName("ID"), WithDetail("(y)"),
                                              Children(WithName("y"))))))),
      },
      {
          R"cpp(
            // More typical example where macro only generates part of a decl.
            #define TEST(A, B) class A##_##B { void go(); }; void A##_##B::go()
            TEST(DocumentSymbols, Macro) { }
          )cpp",
          AllOf(WithName("TEST"), WithDetail("(DocumentSymbols, Macro)"),
                Children(AllOf(WithName("DocumentSymbols_Macro"),
                               Children(WithName("go"))),
                         WithName("DocumentSymbols_Macro::go"))),
      },
      {
          R"cpp(
            // Nested macros.
            #define NAMESPACE(NS, BODY) namespace NS { BODY }
            NAMESPACE(a, NAMESPACE(b, int x;))
          )cpp",
          AllOf(
              WithName("NAMESPACE"), WithDetail("(a, NAMESPACE(b, int x;))"),
              Children(AllOf(
                  WithName("a"),
                  Children(AllOf(WithName("NAMESPACE"),
                                 // FIXME: nested expansions not in TokenBuffer
                                 WithDetail(""),
                                 Children(AllOf(WithName("b"),
                                                Children(WithName("x"))))))))),
      },
      {
          R"cpp(
            // Macro invoked from body is not exposed.
            #define INNER(X) int X
            #define OUTER(X) INNER(X)
            OUTER(foo);
          )cpp",
          AllOf(WithName("OUTER"), WithDetail("(foo)"),
                Children(WithName("foo"))),
      },
  };
  for (const Test &T : Tests) {
    auto TU = TestTU::withCode(T.Code);
    EXPECT_THAT(getSymbols(TU.build()), ElementsAre(T.Matcher)) << T.Code;
  }
}

TEST(DocumentSymbols, RangeFromMacro) {
  TestTU TU;
  Annotations Main(R"(
    #define FF(name) \
      class name##_Test {};

    $expansion1[[FF]](abc);

    #define FF2() \
      class Test {}

    $expansion2parens[[$expansion2[[FF2]]()]];

    #define FF3() \
      void waldo()

    $fullDef[[FF3() {
      int var = 42;
    }]]
  )");
  TU.Code = Main.code().str();
  EXPECT_THAT(
      getSymbols(TU.build()),
      ElementsAre(
          AllOf(WithName("FF"), WithDetail("(abc)"),
                Children(AllOf(WithName("abc_Test"), WithDetail("class"),
                               SymNameRange(Main.range("expansion1"))))),
          AllOf(WithName("FF2"), WithDetail("()"),
                SymNameRange(Main.range("expansion2")),
                SymRange(Main.range("expansion2parens")),
                Children(AllOf(WithName("Test"), WithDetail("class"),
                               SymNameRange(Main.range("expansion2"))))),
          AllOf(WithName("FF3"), WithDetail("()"),
                SymRange(Main.range("fullDef")),
                Children(AllOf(WithName("waldo"), WithDetail("void ()"),
                               SymRange(Main.range("fullDef")))))));
}

TEST(DocumentSymbols, FuncTemplates) {
  TestTU TU;
  Annotations Source(R"cpp(
    template <class T>
    T foo() {}

    auto x = foo<int>();
    auto y = foo<double>();
  )cpp");
  TU.Code = Source.code().str();
  // Make sure we only see the template declaration, not instantiations.
  EXPECT_THAT(getSymbols(TU.build()),
              ElementsAre(AllOf(WithName("foo"), WithDetail("template T ()")),
                          AllOf(WithName("x"), WithDetail("int")),
                          AllOf(WithName("y"), WithDetail("double"))));
}

TEST(DocumentSymbols, UsingDirectives) {
  TestTU TU;
  Annotations Source(R"cpp(
    namespace ns {
      int foo;
    }

    namespace ns_alias = ns;

    using namespace ::ns;     // check we don't loose qualifiers.
    using namespace ns_alias; // and namespace aliases.
  )cpp");
  TU.Code = Source.code().str();
  EXPECT_THAT(getSymbols(TU.build()),
              ElementsAre(WithName("ns"), WithName("ns_alias"),
                          WithName("using namespace ::ns"),
                          WithName("using namespace ns_alias")));
}

TEST(DocumentSymbols, TempSpecs) {
  TestTU TU;
  TU.Code = R"cpp(
      template <typename T, typename U, int X = 5> class Foo {};
      template <typename T> class Foo<int, T> {};
      template <> class Foo<bool, int> {};
      template <> class Foo<bool, int, 3> {};
      )cpp";
  // Foo is higher ranked because of exact name match.
  EXPECT_THAT(getSymbols(TU.build()),
              UnorderedElementsAre(
                  AllOf(WithName("Foo"), WithKind(SymbolKind::Class),
                        WithDetail("template class")),
                  AllOf(WithName("Foo<int, T>"), WithKind(SymbolKind::Class),
                        WithDetail("template class")),
                  AllOf(WithName("Foo<bool, int>"), WithKind(SymbolKind::Class),
                        WithDetail("class")),
                  AllOf(WithName("Foo<bool, int, 3>"),
                        WithKind(SymbolKind::Class), WithDetail("class"))));
}

TEST(DocumentSymbols, Qualifiers) {
  TestTU TU;
  TU.Code = R"cpp(
    namespace foo { namespace bar {
      struct Cls;

      int func1();
      int func2();
      int func3();
      int func4();
    }}

    struct foo::bar::Cls { };

    int foo::bar::func1() { return 10; }
    int ::foo::bar::func2() { return 20; }

    using namespace foo;
    int bar::func3() { return 30; }

    namespace alias = foo::bar;
    int ::alias::func4() { return 40; }
  )cpp";

  // All the qualifiers should be preserved exactly as written.
  EXPECT_THAT(getSymbols(TU.build()),
              UnorderedElementsAre(
                  WithName("foo"), WithName("foo::bar::Cls"),
                  WithName("foo::bar::func1"), WithName("::foo::bar::func2"),
                  WithName("using namespace foo"), WithName("bar::func3"),
                  WithName("alias"), WithName("::alias::func4")));
}

TEST(DocumentSymbols, QualifiersWithTemplateArgs) {
  TestTU TU;
  TU.Code = R"cpp(
      template <typename T, typename U = double> class Foo;

      template <>
      class Foo<int, double> {
        int method1();
        int method2();
        int method3();
      };

      using int_type = int;

      // Typedefs should be preserved!
      int Foo<int_type, double>::method1() { return 10; }

      // Default arguments should not be shown!
      int Foo<int>::method2() { return 20; }

      using Foo_type = Foo<int>;
      // If the whole type is aliased, this should be preserved too!
      int Foo_type::method3() { return 30; }
      )cpp";
  EXPECT_THAT(getSymbols(TU.build()),
              UnorderedElementsAre(
                  AllOf(WithName("Foo"), WithDetail("template class")),
                  AllOf(WithName("Foo<int, double>"), WithDetail("class")),
                  AllOf(WithName("int_type"), WithDetail("type alias")),
                  AllOf(WithName("Foo<int_type, double>::method1"),
                        WithDetail("int ()")),
                  AllOf(WithName("Foo<int>::method2"), WithDetail("int ()")),
                  AllOf(WithName("Foo_type"), WithDetail("type alias")),
                  AllOf(WithName("Foo_type::method3"), WithDetail("int ()"))));
}

TEST(DocumentSymbolsTest, Ranges) {
  TestTU TU;
  Annotations Main(R"(
      $foo[[int foo(bool Argument) {
        return 42;
      }]]

      $variable[[char GLOBAL_VARIABLE]];

      $ns[[namespace ns {
      $bar[[class Bar {
      public:
        $ctor[[Bar() {}]]
        $dtor[[~Bar()]];

      private:
        $field[[unsigned Baz]];

        $getbaz[[unsigned getBaz() { return Baz; }]]
      }]];
      }]] // namespace ns

      $forwardclass[[class ForwardClassDecl]];

      $struct[[struct StructDefinition {
        $structfield[[int *Pointer = nullptr]];
      }]];
      $forwardstruct[[struct StructDeclaration]];

      $forwardfunc[[void forwardFunctionDecl(int Something)]];
    )");
  TU.Code = Main.code().str();
  EXPECT_THAT(
      getSymbols(TU.build()),
      UnorderedElementsAre(
          AllOf(WithName("foo"), WithKind(SymbolKind::Function),
                WithDetail("int (bool)"), SymRange(Main.range("foo"))),
          AllOf(WithName("GLOBAL_VARIABLE"), WithKind(SymbolKind::Variable),
                WithDetail("char"), SymRange(Main.range("variable"))),
          AllOf(
              WithName("ns"), WithKind(SymbolKind::Namespace),
              SymRange(Main.range("ns")),
              Children(AllOf(
                  WithName("Bar"), WithKind(SymbolKind::Class),
                  WithDetail("class"), SymRange(Main.range("bar")),
                  Children(
                      AllOf(WithName("Bar"), WithKind(SymbolKind::Constructor),
                            WithDetail("()"), SymRange(Main.range("ctor"))),
                      AllOf(WithName("~Bar"), WithKind(SymbolKind::Constructor),
                            WithDetail(""), SymRange(Main.range("dtor"))),
                      AllOf(WithName("Baz"), WithKind(SymbolKind::Field),
                            WithDetail("unsigned int"),
                            SymRange(Main.range("field"))),
                      AllOf(WithName("getBaz"), WithKind(SymbolKind::Method),
                            WithDetail("unsigned int ()"),
                            SymRange(Main.range("getbaz"))))))),
          AllOf(WithName("ForwardClassDecl"), WithKind(SymbolKind::Class),
                WithDetail("class"), SymRange(Main.range("forwardclass"))),
          AllOf(WithName("StructDefinition"), WithKind(SymbolKind::Struct),
                WithDetail("struct"), SymRange(Main.range("struct")),
                Children(AllOf(WithName("Pointer"), WithKind(SymbolKind::Field),
                               WithDetail("int *"),
                               SymRange(Main.range("structfield"))))),
          AllOf(WithName("StructDeclaration"), WithKind(SymbolKind::Struct),
                WithDetail("struct"), SymRange(Main.range("forwardstruct"))),
          AllOf(WithName("forwardFunctionDecl"), WithKind(SymbolKind::Function),
                WithDetail("void (int)"),
                SymRange(Main.range("forwardfunc")))));
}

TEST(DocumentSymbolsTest, DependentType) {
  TestTU TU;
  TU.Code = R"(
    template <typename T> auto plus(T x, T y) -> decltype(x + y) { return x + y; }

    template <typename Key, typename Value> class Pair {};

    template <typename Key, typename Value>
    struct Context : public Pair<Key, Value> {
      using Pair<Key, Value>::Pair;
    };
    )";
  EXPECT_THAT(
      getSymbols(TU.build()),
      ElementsAre(
          AllOf(WithName("plus"),
                WithDetail("template auto (T, T) -> decltype(x + y)")),
          AllOf(WithName("Pair"), WithDetail("template class")),
          AllOf(WithName("Context"), WithDetail("template struct"),
                Children(AllOf(
                    WithName("Pair<type-parameter-0-0, type-parameter-0-1>"),
                    WithDetail("<dependent type>"))))));
}

TEST(DocumentSymbolsTest, ObjCCategoriesAndClassExtensions) {
  TestTU TU;
  TU.ExtraArgs = {"-xobjective-c++", "-Wno-objc-root-class"};
  Annotations Main(R"cpp(
      $Cat[[@interface Cat
      + (id)sharedCat;
      @end]]
      $SneakyCat[[@interface Cat (Sneaky)
      - (id)sneak:(id)behavior;
      @end]]

      $MeowCat[[@interface Cat ()
      - (void)meow;
      @end]]
      $PurCat[[@interface Cat ()
      - (void)pur;
      @end]]
    )cpp");
  TU.Code = Main.code().str();
  EXPECT_THAT(
      getSymbols(TU.build()),
      ElementsAre(
          AllOf(WithName("Cat"), SymRange(Main.range("Cat")),
                Children(AllOf(WithName("+sharedCat"),
                               WithKind(SymbolKind::Method)))),
          AllOf(WithName("Cat(Sneaky)"), SymRange(Main.range("SneakyCat")),
                Children(
                    AllOf(WithName("-sneak:"), WithKind(SymbolKind::Method)))),
          AllOf(
              WithName("Cat()"), SymRange(Main.range("MeowCat")),
              Children(AllOf(WithName("-meow"), WithKind(SymbolKind::Method)))),
          AllOf(WithName("Cat()"), SymRange(Main.range("PurCat")),
                Children(
                    AllOf(WithName("-pur"), WithKind(SymbolKind::Method))))));
}

} // namespace
} // namespace clangd
} // namespace clang
