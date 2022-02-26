//===-- FindSymbolsTests.cpp -------------------------*- C++ -*------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Annotations.h"
#include "FindSymbols.h"
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
MATCHER_P(qName, Name, "") {
  if (arg.containerName.empty())
    return arg.name == Name;
  return (arg.containerName + "::" + arg.name) == Name;
}
MATCHER_P(withName, N, "") { return arg.name == N; }
MATCHER_P(withKind, Kind, "") { return arg.kind == Kind; }
MATCHER_P(withDetail, Detail, "") { return arg.detail == Detail; }
MATCHER_P(symRange, Range, "") { return arg.range == Range; }

// GMock helpers for matching DocumentSymbol.
MATCHER_P(symNameRange, Range, "") { return arg.selectionRange == Range; }
template <class... ChildMatchers>
::testing::Matcher<DocumentSymbol> children(ChildMatchers... ChildrenM) {
  return Field(&DocumentSymbol::children, UnorderedElementsAre(ChildrenM...));
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
              ElementsAre(AllOf(qName("MACRO"), withKind(SymbolKind::String))));
}

TEST(WorkspaceSymbols, NoLocals) {
  TestTU TU;
  TU.Code = R"cpp(
      void test(int FirstParam, int SecondParam) {
        struct LocalClass {};
        int local_var;
      })cpp";
  EXPECT_THAT(getSymbols(TU, "l"), ElementsAre(qName("LocalClass")));
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
                  AllOf(qName("GlobalStruct"), withKind(SymbolKind::Struct)),
                  AllOf(qName("global_func"), withKind(SymbolKind::Function)),
                  AllOf(qName("global_var"), withKind(SymbolKind::Variable))));
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
              ElementsAre(AllOf(qName("UnnamedStruct"),
                                withKind(SymbolKind::Variable))));
  EXPECT_THAT(getSymbols(TU, "InUnnamed"),
              ElementsAre(AllOf(qName("(anonymous struct)::InUnnamed"),
                                withKind(SymbolKind::Field))));
}

TEST(WorkspaceSymbols, InMainFile) {
  TestTU TU;
  TU.Code = R"cpp(
      int test() {}
      static void test2() {}
      )cpp";
  EXPECT_THAT(getSymbols(TU, "test"),
              ElementsAre(qName("test"), qName("test2")));
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
                  qName("ans1"), qName("ans1::ai1"), qName("ans1::ans2"),
                  qName("ans1::ans2::ai2"), qName("ans1::ans2::ans3"),
                  qName("ans1::ans2::ans3::ai3")));
  EXPECT_THAT(getSymbols(TU, "::"), ElementsAre(qName("ans1")));
  EXPECT_THAT(getSymbols(TU, "::a"), ElementsAre(qName("ans1")));
  EXPECT_THAT(getSymbols(TU, "ans1::"),
              UnorderedElementsAre(qName("ans1::ai1"), qName("ans1::ans2"),
                                   qName("ans1::ans2::ai2"),
                                   qName("ans1::ans2::ans3"),
                                   qName("ans1::ans2::ans3::ai3")));
  EXPECT_THAT(getSymbols(TU, "ans2::"),
              UnorderedElementsAre(qName("ans1::ans2::ai2"),
                                   qName("ans1::ans2::ans3"),
                                   qName("ans1::ans2::ans3::ai3")));
  EXPECT_THAT(getSymbols(TU, "::ans1"), ElementsAre(qName("ans1")));
  EXPECT_THAT(getSymbols(TU, "::ans1::"),
              UnorderedElementsAre(qName("ans1::ai1"), qName("ans1::ans2")));
  EXPECT_THAT(getSymbols(TU, "::ans1::ans2"), ElementsAre(qName("ans1::ans2")));
  EXPECT_THAT(getSymbols(TU, "::ans1::ans2::"),
              ElementsAre(qName("ans1::ans2::ai2"), qName("ans1::ans2::ans3")));

  // Ensure sub-sequence matching works.
  EXPECT_THAT(getSymbols(TU, "ans1::ans3::ai"),
              UnorderedElementsAre(qName("ans1::ans2::ans3::ai3")));
}

TEST(WorkspaceSymbols, AnonymousNamespace) {
  TestTU TU;
  TU.Code = R"cpp(
      namespace {
      void test() {}
      }
      )cpp";
  EXPECT_THAT(getSymbols(TU, "test"), ElementsAre(qName("test")));
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
              UnorderedElementsAre(qName("foo"), qName("foo2")));
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
                  AllOf(qName("Foo"), withKind(SymbolKind::Class)),
                  AllOf(qName("foo"), withKind(SymbolKind::Function)),
                  AllOf(qName("ns"), withKind(SymbolKind::Namespace))));
  EXPECT_THAT(getSymbols(TU, ":"), IsEmpty());
  EXPECT_THAT(getSymbols(TU, ""),
              UnorderedElementsAre(qName("foo"), qName("Foo"), qName("Foo::a"),
                                   qName("ns"), qName("ns::foo2")));
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
  EXPECT_THAT(getSymbols(TU, "Red"), ElementsAre(qName("Red")));
  EXPECT_THAT(getSymbols(TU, "::Red"), ElementsAre(qName("Red")));
  EXPECT_THAT(getSymbols(TU, "Green"), ElementsAre(qName("Green")));
  EXPECT_THAT(getSymbols(TU, "Green"), ElementsAre(qName("Green")));
  EXPECT_THAT(getSymbols(TU, "Color2::Yellow"),
              ElementsAre(qName("Color2::Yellow")));
  EXPECT_THAT(getSymbols(TU, "Yellow"), ElementsAre(qName("Color2::Yellow")));

  EXPECT_THAT(getSymbols(TU, "ns::Black"), ElementsAre(qName("ns::Black")));
  EXPECT_THAT(getSymbols(TU, "ns::Blue"), ElementsAre(qName("ns::Blue")));
  EXPECT_THAT(getSymbols(TU, "ns::Color4::White"),
              ElementsAre(qName("ns::Color4::White")));
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
  EXPECT_THAT(getSymbols(TU, "::"), ElementsAre(qName("func"), qName("ns")));
}

TEST(WorkspaceSymbols, RankingPartialNamespace) {
  TestTU TU;
  TU.Code = R"cpp(
    namespace ns1 {
      namespace ns2 { struct Foo {}; }
    }
    namespace ns2 { struct FooB {}; })cpp";
  EXPECT_THAT(getSymbols(TU, "ns2::f"),
              ElementsAre(qName("ns2::FooB"), qName("ns1::ns2::Foo")));
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
                  AllOf(qName("foo"), withKind(SymbolKind::Variable)),
                  AllOf(qName("foo2"), withKind(SymbolKind::Variable))));

  EXPECT_THAT(getSymbols(TU, "foo", 1), ElementsAre(qName("foo")));
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
          AllOf(qName("Foo"), withKind(SymbolKind::Class)),
          AllOf(qName("Foo<int, T>"), withKind(SymbolKind::Class)),
          AllOf(qName("Foo<bool, int>"), withKind(SymbolKind::Class)),
          AllOf(qName("Foo<bool, int, 3>"), withKind(SymbolKind::Class))));
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
          {AllOf(withName("Foo"), withKind(SymbolKind::Class),
                 withDetail("class"), children()),
           AllOf(withName("Foo"), withKind(SymbolKind::Class),
                 withDetail("class"),
                 children(
                     AllOf(withName("Foo"), withKind(SymbolKind::Constructor),
                           withDetail("()"), children()),
                     AllOf(withName("Foo"), withKind(SymbolKind::Constructor),
                           withDetail("(int)"), children()),
                     AllOf(withName("f"), withKind(SymbolKind::Method),
                           withDetail("void ()"), children()),
                     AllOf(withName("operator="), withKind(SymbolKind::Method),
                           withDetail("Foo &(const Foo &)"), children()),
                     AllOf(withName("~Foo"), withKind(SymbolKind::Constructor),
                           withDetail(""), children()),
                     AllOf(withName("Nested"), withKind(SymbolKind::Class),
                           withDetail("class"),
                           children(AllOf(
                               withName("f"), withKind(SymbolKind::Method),
                               withDetail("void ()"), children()))))),
           AllOf(withName("Friend"), withKind(SymbolKind::Class),
                 withDetail("class"), children()),
           AllOf(withName("f1"), withKind(SymbolKind::Function),
                 withDetail("void ()"), children()),
           AllOf(withName("f2"), withKind(SymbolKind::Function),
                 withDetail("void ()"), children()),
           AllOf(withName("KInt"), withKind(SymbolKind::Variable),
                 withDetail("const int"), children()),
           AllOf(withName("kStr"), withKind(SymbolKind::Variable),
                 withDetail("const char *"), children()),
           AllOf(withName("f1"), withKind(SymbolKind::Function),
                 withDetail("void ()"), children()),
           AllOf(
               withName("foo"), withKind(SymbolKind::Namespace), withDetail(""),
               children(AllOf(withName("int32"), withKind(SymbolKind::Class),
                              withDetail("type alias"), children()),
                        AllOf(withName("int32_t"), withKind(SymbolKind::Class),
                              withDetail("type alias"), children()),
                        AllOf(withName("v1"), withKind(SymbolKind::Variable),
                              withDetail("int"), children()),
                        AllOf(withName("bar"), withKind(SymbolKind::Namespace),
                              withDetail(""),
                              children(AllOf(withName("v2"),
                                             withKind(SymbolKind::Variable),
                                             withDetail("int"), children()))),
                        AllOf(withName("baz"), withKind(SymbolKind::Namespace),
                              withDetail(""), children()),
                        AllOf(withName("v2"), withKind(SymbolKind::Namespace),
                              withDetail(""))))}));
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
          AllOf(withName("Foo"), withKind(SymbolKind::Class),
                withDetail("class"),
                children(AllOf(withName("f"), withKind(SymbolKind::Method),
                               withDetail("void ()"),
                               symNameRange(Main.range("decl"))))),
          AllOf(withName("Foo::f"), withKind(SymbolKind::Method),
                withDetail("void ()"), symNameRange(Main.range("def")))));
}

TEST(DocumentSymbols, Concepts) {
  TestTU TU;
  TU.ExtraArgs = {"-std=c++20"};
  TU.Code = "template <typename T> concept C = requires(T t) { t.foo(); };";

  EXPECT_THAT(getSymbols(TU.build()),
              ElementsAre(AllOf(withName("C"), withDetail("concept"))));
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
              ElementsAre(withName("foo"), withName("Foo"),
                          AllOf(withName("ns"),
                                children(withName("bar"), withName("Bar")))));
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
              ElementsAre(withName("foo"), withName("Foo")));
}

TEST(DocumentSymbols, NoLocals) {
  TestTU TU;
  TU.Code = R"cpp(
      void test(int FirstParam, int SecondParam) {
        struct LocalClass {};
        int local_var;
      })cpp";
  EXPECT_THAT(getSymbols(TU.build()), ElementsAre(withName("test")));
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
      ElementsAre(AllOf(withName("(anonymous struct)"),
                        withKind(SymbolKind::Struct), withDetail("struct"),
                        children(AllOf(withName("InUnnamed"),
                                       withKind(SymbolKind::Field),
                                       withDetail("int"), children()))),
                  AllOf(withName("UnnamedStruct"),
                        withKind(SymbolKind::Variable),
                        withDetail("struct (unnamed)"), children())));
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
              ElementsAre(withName("i"), withName("test")));
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
          AllOf(withName("Tmpl"), withKind(SymbolKind::Struct),
                withDetail("template struct"),
                children(AllOf(withName("x"), withKind(SymbolKind::Field),
                               withDetail("T")))),
          AllOf(withName("Tmpl<int>"), withKind(SymbolKind::Struct),
                withDetail("struct"),
                children(AllOf(withName("y"), withDetail("int")))),
          AllOf(withName("Tmpl<float>"), withKind(SymbolKind::Struct),
                withDetail("struct"), children()),
          AllOf(withName("Tmpl<double>"), withKind(SymbolKind::Struct),
                withDetail("struct"), children()),
          AllOf(withName("funcTmpl"), withDetail("template int (U)"),
                children()),
          AllOf(withName("funcTmpl<int>"), withDetail("int (double)"),
                children()),
          AllOf(withName("varTmpl"), withDetail("template int"), children()),
          AllOf(withName("varTmpl<int>"), withDetail("double"), children())));
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
          {AllOf(withName("ans1"),
                 children(AllOf(withName("ai1"), children()),
                          AllOf(withName("ans2"), children(withName("ai2"))))),
           AllOf(withName("(anonymous namespace)"), children(withName("test"))),
           AllOf(withName("na"),
                 children(AllOf(withName("nb"), children(withName("Foo"))))),
           AllOf(withName("na"),
                 children(AllOf(withName("nb"), children(withName("Bar")))))}));
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
          AllOf(withName("(anonymous enum)"), withDetail("enum"),
                children(AllOf(withName("Red"), withDetail("(unnamed)")))),
          AllOf(withName("Color"), withDetail("enum"),
                children(AllOf(withName("Green"), withDetail("Color")))),
          AllOf(withName("Color2"), withDetail("enum"),
                children(AllOf(withName("Yellow"), withDetail("Color2")))),
          AllOf(withName("ns"),
                children(AllOf(withName("(anonymous enum)"), withDetail("enum"),
                               children(AllOf(withName("Black"),
                                              withDetail("(unnamed)"))))))));
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
          AllOf(withName("DEFINE_FLAG"), withDetail("(pretty)"),
                children(withName("FLAGS_pretty"), withName("FLAGS_nopretty"))),
      },
      {
          R"cpp(
            // Hierarchy is determined by primary (name) location.
            #define ID(X) X
            namespace ID(ns) { int ID(y); }
          )cpp",
          AllOf(withName("ID"), withDetail("(ns)"),
                children(AllOf(withName("ns"),
                               children(AllOf(withName("ID"), withDetail("(y)"),
                                              children(withName("y"))))))),
      },
      {
          R"cpp(
            // More typical example where macro only generates part of a decl.
            #define TEST(A, B) class A##_##B { void go(); }; void A##_##B::go()
            TEST(DocumentSymbols, Macro) { }
          )cpp",
          AllOf(withName("TEST"), withDetail("(DocumentSymbols, Macro)"),
                children(AllOf(withName("DocumentSymbols_Macro"),
                               children(withName("go"))),
                         withName("DocumentSymbols_Macro::go"))),
      },
      {
          R"cpp(
            // Nested macros.
            #define NAMESPACE(NS, BODY) namespace NS { BODY }
            NAMESPACE(a, NAMESPACE(b, int x;))
          )cpp",
          AllOf(
              withName("NAMESPACE"), withDetail("(a, NAMESPACE(b, int x;))"),
              children(AllOf(
                  withName("a"),
                  children(AllOf(withName("NAMESPACE"),
                                 // FIXME: nested expansions not in TokenBuffer
                                 withDetail(""),
                                 children(AllOf(withName("b"),
                                                children(withName("x"))))))))),
      },
      {
          R"cpp(
            // Macro invoked from body is not exposed.
            #define INNER(X) int X
            #define OUTER(X) INNER(X)
            OUTER(foo);
          )cpp",
          AllOf(withName("OUTER"), withDetail("(foo)"),
                children(withName("foo"))),
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
          AllOf(withName("FF"), withDetail("(abc)"),
                children(AllOf(withName("abc_Test"), withDetail("class"),
                               symNameRange(Main.range("expansion1"))))),
          AllOf(withName("FF2"), withDetail("()"),
                symNameRange(Main.range("expansion2")),
                symRange(Main.range("expansion2parens")),
                children(AllOf(withName("Test"), withDetail("class"),
                               symNameRange(Main.range("expansion2"))))),
          AllOf(withName("FF3"), withDetail("()"),
                symRange(Main.range("fullDef")),
                children(AllOf(withName("waldo"), withDetail("void ()"),
                               symRange(Main.range("fullDef")))))));
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
              ElementsAre(AllOf(withName("foo"), withDetail("template T ()")),
                          AllOf(withName("x"), withDetail("int")),
                          AllOf(withName("y"), withDetail("double"))));
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
              ElementsAre(withName("ns"), withName("ns_alias"),
                          withName("using namespace ::ns"),
                          withName("using namespace ns_alias")));
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
                  AllOf(withName("Foo"), withKind(SymbolKind::Class),
                        withDetail("template class")),
                  AllOf(withName("Foo<int, T>"), withKind(SymbolKind::Class),
                        withDetail("template class")),
                  AllOf(withName("Foo<bool, int>"), withKind(SymbolKind::Class),
                        withDetail("class")),
                  AllOf(withName("Foo<bool, int, 3>"),
                        withKind(SymbolKind::Class), withDetail("class"))));
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
                  withName("foo"), withName("foo::bar::Cls"),
                  withName("foo::bar::func1"), withName("::foo::bar::func2"),
                  withName("using namespace foo"), withName("bar::func3"),
                  withName("alias"), withName("::alias::func4")));
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
                  AllOf(withName("Foo"), withDetail("template class")),
                  AllOf(withName("Foo<int, double>"), withDetail("class")),
                  AllOf(withName("int_type"), withDetail("type alias")),
                  AllOf(withName("Foo<int_type, double>::method1"),
                        withDetail("int ()")),
                  AllOf(withName("Foo<int>::method2"), withDetail("int ()")),
                  AllOf(withName("Foo_type"), withDetail("type alias")),
                  AllOf(withName("Foo_type::method3"), withDetail("int ()"))));
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
          AllOf(withName("foo"), withKind(SymbolKind::Function),
                withDetail("int (bool)"), symRange(Main.range("foo"))),
          AllOf(withName("GLOBAL_VARIABLE"), withKind(SymbolKind::Variable),
                withDetail("char"), symRange(Main.range("variable"))),
          AllOf(
              withName("ns"), withKind(SymbolKind::Namespace),
              symRange(Main.range("ns")),
              children(AllOf(
                  withName("Bar"), withKind(SymbolKind::Class),
                  withDetail("class"), symRange(Main.range("bar")),
                  children(
                      AllOf(withName("Bar"), withKind(SymbolKind::Constructor),
                            withDetail("()"), symRange(Main.range("ctor"))),
                      AllOf(withName("~Bar"), withKind(SymbolKind::Constructor),
                            withDetail(""), symRange(Main.range("dtor"))),
                      AllOf(withName("Baz"), withKind(SymbolKind::Field),
                            withDetail("unsigned int"),
                            symRange(Main.range("field"))),
                      AllOf(withName("getBaz"), withKind(SymbolKind::Method),
                            withDetail("unsigned int ()"),
                            symRange(Main.range("getbaz"))))))),
          AllOf(withName("ForwardClassDecl"), withKind(SymbolKind::Class),
                withDetail("class"), symRange(Main.range("forwardclass"))),
          AllOf(withName("StructDefinition"), withKind(SymbolKind::Struct),
                withDetail("struct"), symRange(Main.range("struct")),
                children(AllOf(withName("Pointer"), withKind(SymbolKind::Field),
                               withDetail("int *"),
                               symRange(Main.range("structfield"))))),
          AllOf(withName("StructDeclaration"), withKind(SymbolKind::Struct),
                withDetail("struct"), symRange(Main.range("forwardstruct"))),
          AllOf(withName("forwardFunctionDecl"), withKind(SymbolKind::Function),
                withDetail("void (int)"),
                symRange(Main.range("forwardfunc")))));
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
          AllOf(withName("plus"),
                withDetail("template auto (T, T) -> decltype(x + y)")),
          AllOf(withName("Pair"), withDetail("template class")),
          AllOf(withName("Context"), withDetail("template struct"),
                children(AllOf(
                    withName("Pair<type-parameter-0-0, type-parameter-0-1>"),
                    withDetail("<dependent type>"))))));
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
          AllOf(withName("Cat"), symRange(Main.range("Cat")),
                children(AllOf(withName("+sharedCat"),
                               withKind(SymbolKind::Method)))),
          AllOf(withName("Cat(Sneaky)"), symRange(Main.range("SneakyCat")),
                children(
                    AllOf(withName("-sneak:"), withKind(SymbolKind::Method)))),
          AllOf(
              withName("Cat()"), symRange(Main.range("MeowCat")),
              children(AllOf(withName("-meow"), withKind(SymbolKind::Method)))),
          AllOf(withName("Cat()"), symRange(Main.range("PurCat")),
                children(
                    AllOf(withName("-pur"), withKind(SymbolKind::Method))))));
}

TEST(DocumentSymbolsTest, PragmaMarkGroups) {
  TestTU TU;
  TU.ExtraArgs = {"-xobjective-c++", "-Wno-objc-root-class"};
  Annotations Main(R"cpp(
      $DogDef[[@interface Dog
      @end]]

      $DogImpl[[@implementation Dog

      + (id)sharedDoggo { return 0; }

      #pragma $Overrides[[mark - Overrides

      - (id)init {
        return self;
      }
      - (void)bark {}]]

      #pragma $Specifics[[mark - Dog Specifics

      - (int)isAGoodBoy {
        return 1;
      }]]
      @]]end  // FIXME: Why doesn't this include the 'end'?

      #pragma $End[[mark - End
]]
    )cpp");
  TU.Code = Main.code().str();
  EXPECT_THAT(
      getSymbols(TU.build()),
      UnorderedElementsAre(
          AllOf(withName("Dog"), symRange(Main.range("DogDef"))),
          AllOf(withName("Dog"), symRange(Main.range("DogImpl")),
                children(AllOf(withName("+sharedDoggo"),
                               withKind(SymbolKind::Method)),
                         AllOf(withName("Overrides"),
                               symRange(Main.range("Overrides")),
                               children(AllOf(withName("-init"),
                                              withKind(SymbolKind::Method)),
                                        AllOf(withName("-bark"),
                                              withKind(SymbolKind::Method)))),
                         AllOf(withName("Dog Specifics"),
                               symRange(Main.range("Specifics")),
                               children(AllOf(withName("-isAGoodBoy"),
                                              withKind(SymbolKind::Method)))))),
          AllOf(withName("End"), symRange(Main.range("End")))));
}

TEST(DocumentSymbolsTest, PragmaMarkGroupsNesting) {
  TestTU TU;
  TU.ExtraArgs = {"-xobjective-c++", "-Wno-objc-root-class"};
  Annotations Main(R"cpp(
      #pragma mark - Foo
      struct Foo {
        #pragma mark - Bar
        void bar() {
           #pragma mark - NotTopDecl
        }
      };
      void bar() {}
    )cpp");
  TU.Code = Main.code().str();
  EXPECT_THAT(
      getSymbols(TU.build()),
      UnorderedElementsAre(AllOf(
          withName("Foo"),
          children(AllOf(withName("Foo"),
                         children(AllOf(withName("Bar"),
                                        children(AllOf(withName("bar"),
                                                       children(withName(
                                                           "NotTopDecl"))))))),
                   withName("bar")))));
}

TEST(DocumentSymbolsTest, PragmaMarkGroupsNoNesting) {
  TestTU TU;
  TU.ExtraArgs = {"-xobjective-c++", "-Wno-objc-root-class"};
  Annotations Main(R"cpp(
      #pragma mark Helpers
      void helpA(id obj) {}

      #pragma mark -
      #pragma mark Core

      void coreMethod() {}
    )cpp");
  TU.Code = Main.code().str();
  EXPECT_THAT(getSymbols(TU.build()),
              UnorderedElementsAre(withName("Helpers"), withName("helpA"),
                                   withName("(unnamed group)"),
                                   withName("Core"), withName("coreMethod")));
}

} // namespace
} // namespace clangd
} // namespace clang
