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
  EXPECT_THAT(getSymbols(TU, "l"), IsEmpty());
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
  EXPECT_THAT(getSymbols(TU, ""), IsEmpty());
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
          {AllOf(WithName("Foo"), WithKind(SymbolKind::Class), Children()),
           AllOf(WithName("Foo"), WithKind(SymbolKind::Class),
                 Children(AllOf(WithName("Foo"),
                                WithKind(SymbolKind::Constructor), Children()),
                          AllOf(WithName("Foo"),
                                WithKind(SymbolKind::Constructor), Children()),
                          AllOf(WithName("f"), WithKind(SymbolKind::Method),
                                Children()),
                          AllOf(WithName("operator="),
                                WithKind(SymbolKind::Method), Children()),
                          AllOf(WithName("~Foo"),
                                WithKind(SymbolKind::Constructor), Children()),
                          AllOf(WithName("Nested"), WithKind(SymbolKind::Class),
                                Children(AllOf(WithName("f"),
                                               WithKind(SymbolKind::Method),
                                               Children()))))),
           AllOf(WithName("Friend"), WithKind(SymbolKind::Class), Children()),
           AllOf(WithName("f1"), WithKind(SymbolKind::Function), Children()),
           AllOf(WithName("f2"), WithKind(SymbolKind::Function), Children()),
           AllOf(WithName("KInt"), WithKind(SymbolKind::Variable), Children()),
           AllOf(WithName("kStr"), WithKind(SymbolKind::Variable), Children()),
           AllOf(WithName("f1"), WithKind(SymbolKind::Function), Children()),
           AllOf(
               WithName("foo"), WithKind(SymbolKind::Namespace),
               Children(
                   AllOf(WithName("int32"), WithKind(SymbolKind::Class),
                         Children()),
                   AllOf(WithName("int32_t"), WithKind(SymbolKind::Class),
                         Children()),
                   AllOf(WithName("v1"), WithKind(SymbolKind::Variable),
                         Children()),
                   AllOf(WithName("bar"), WithKind(SymbolKind::Namespace),
                         Children(AllOf(WithName("v2"),
                                        WithKind(SymbolKind::Variable),
                                        Children()))),
                   AllOf(WithName("baz"), WithKind(SymbolKind::Namespace),
                         Children()),
                   AllOf(WithName("v2"), WithKind(SymbolKind::Namespace))))}));
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
                Children(AllOf(WithName("f"), WithKind(SymbolKind::Method),
                               SymNameRange(Main.range("decl"))))),
          AllOf(WithName("Foo::f"), WithKind(SymbolKind::Method),
                SymNameRange(Main.range("def")))));
}

TEST(DocumentSymbols, Concepts) {
  TestTU TU;
  TU.ExtraArgs = {"-std=c++20"};
  TU.Code = "template <typename T> concept C = requires(T t) { t.foo(); };";

  EXPECT_THAT(getSymbols(TU.build()), ElementsAre(WithName("C")));
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
      ElementsAre(
          AllOf(WithName("(anonymous struct)"), WithKind(SymbolKind::Struct),
                Children(AllOf(WithName("InUnnamed"),
                               WithKind(SymbolKind::Field), Children()))),
          AllOf(WithName("UnnamedStruct"), WithKind(SymbolKind::Variable),
                Children())));
}

TEST(DocumentSymbols, InHeaderFile) {
  TestTU TU;
  TU.AdditionalFiles["bar.h"] = R"cpp(
      int foo() {
      }
      )cpp";
  TU.Code = R"cpp(
      #include "bar.h"
      int test() {
      }
      )cpp";
  EXPECT_THAT(getSymbols(TU.build()), ElementsAre(WithName("test")));
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
                Children(AllOf(WithName("x"), WithKind(SymbolKind::Field)))),
          AllOf(WithName("Tmpl<int>"), WithKind(SymbolKind::Struct),
                Children(WithName("y"))),
          AllOf(WithName("Tmpl<float>"), WithKind(SymbolKind::Struct),
                Children()),
          AllOf(WithName("Tmpl<double>"), WithKind(SymbolKind::Struct),
                Children()),
          AllOf(WithName("funcTmpl"), Children()),
          AllOf(WithName("funcTmpl<int>"), Children()),
          AllOf(WithName("varTmpl"), Children()),
          AllOf(WithName("varTmpl<int>"), Children())));
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
          AllOf(WithName("(anonymous enum)"), Children(WithName("Red"))),
          AllOf(WithName("Color"), Children(WithName("Green"))),
          AllOf(WithName("Color2"), Children(WithName("Yellow"))),
          AllOf(WithName("ns"), Children(AllOf(WithName("(anonymous enum)"),
                                               Children(WithName("Black")))))));
}

TEST(DocumentSymbols, FromMacro) {
  TestTU TU;
  Annotations Main(R"(
    #define FF(name) \
      class name##_Test {};

    $expansion1[[FF]](abc);

    #define FF2() \
      class Test {};

    $expansion2[[FF2]]();

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
          AllOf(WithName("abc_Test"), SymNameRange(Main.range("expansion1"))),
          AllOf(WithName("Test"), SymNameRange(Main.range("expansion2"))),
          AllOf(WithName("waldo"), SymRange(Main.range("fullDef")))));
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
              ElementsAre(WithName("foo"), WithName("x"), WithName("y")));
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
  EXPECT_THAT(
      getSymbols(TU.build()),
      UnorderedElementsAre(
          AllOf(WithName("Foo"), WithKind(SymbolKind::Class)),
          AllOf(WithName("Foo<int, T>"), WithKind(SymbolKind::Class)),
          AllOf(WithName("Foo<bool, int>"), WithKind(SymbolKind::Class)),
          AllOf(WithName("Foo<bool, int, 3>"), WithKind(SymbolKind::Class))));
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
  EXPECT_THAT(
      getSymbols(TU.build()),
      UnorderedElementsAre(WithName("Foo"), WithName("Foo<int, double>"),
                           WithName("int_type"),
                           WithName("Foo<int_type, double>::method1"),
                           WithName("Foo<int>::method2"), WithName("Foo_type"),
                           WithName("Foo_type::method3")));
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
                SymRange(Main.range("foo"))),
          AllOf(WithName("GLOBAL_VARIABLE"), WithKind(SymbolKind::Variable),
                SymRange(Main.range("variable"))),
          AllOf(
              WithName("ns"), WithKind(SymbolKind::Namespace),
              SymRange(Main.range("ns")),
              Children(AllOf(
                  WithName("Bar"), WithKind(SymbolKind::Class),
                  SymRange(Main.range("bar")),
                  Children(
                      AllOf(WithName("Bar"), WithKind(SymbolKind::Constructor),
                            SymRange(Main.range("ctor"))),
                      AllOf(WithName("~Bar"), WithKind(SymbolKind::Constructor),
                            SymRange(Main.range("dtor"))),
                      AllOf(WithName("Baz"), WithKind(SymbolKind::Field),
                            SymRange(Main.range("field"))),
                      AllOf(WithName("getBaz"), WithKind(SymbolKind::Method),
                            SymRange(Main.range("getbaz"))))))),
          AllOf(WithName("ForwardClassDecl"), WithKind(SymbolKind::Class),
                SymRange(Main.range("forwardclass"))),
          AllOf(WithName("StructDefinition"), WithKind(SymbolKind::Struct),
                SymRange(Main.range("struct")),
                Children(AllOf(WithName("Pointer"), WithKind(SymbolKind::Field),
                               SymRange(Main.range("structfield"))))),
          AllOf(WithName("StructDeclaration"), WithKind(SymbolKind::Struct),
                SymRange(Main.range("forwardstruct"))),
          AllOf(WithName("forwardFunctionDecl"), WithKind(SymbolKind::Function),
                SymRange(Main.range("forwardfunc")))));
}

} // namespace
} // namespace clangd
} // namespace clang
