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
MATCHER_P(SymRange, Range, "") { return arg.location.range == Range; }

// GMock helpers for matching DocumentSymbol.
MATCHER_P(SymNameRange, Range, "") { return arg.selectionRange == Range; }
template <class... ChildMatchers>
::testing::Matcher<DocumentSymbol> Children(ChildMatchers... ChildrenM) {
  return Field(&DocumentSymbol::children, ElementsAre(ChildrenM...));
}

ClangdServer::Options optsForTests() {
  auto ServerOpts = ClangdServer::optsForTest();
  ServerOpts.WorkspaceRoot = testRoot();
  ServerOpts.BuildDynamicSymbolIndex = true;
  return ServerOpts;
}

class WorkspaceSymbolsTest : public ::testing::Test {
public:
  WorkspaceSymbolsTest() : Server(CDB, FSProvider, optsForTests()) {
    // Make sure the test root directory is created.
    FSProvider.Files[testPath("unused")] = "";
    CDB.ExtraClangFlags = {"-xc++"};
  }

protected:
  MockFSProvider FSProvider;
  MockCompilationDatabase CDB;
  ClangdServer Server;
  int Limit = 0;

  std::vector<SymbolInformation> getSymbols(llvm::StringRef Query) {
    EXPECT_TRUE(Server.blockUntilIdleForTest()) << "Waiting for preamble";
    auto SymbolInfos = runWorkspaceSymbols(Server, Query, Limit);
    EXPECT_TRUE(bool(SymbolInfos)) << "workspaceSymbols returned an error";
    return *SymbolInfos;
  }

  void addFile(llvm::StringRef FileName, llvm::StringRef Contents) {
    auto Path = testPath(FileName);
    FSProvider.Files[Path] = std::string(Contents);
    Server.addDocument(Path, Contents);
  }
};

} // namespace

TEST_F(WorkspaceSymbolsTest, Macros) {
  addFile("foo.cpp", R"cpp(
       #define MACRO X
       )cpp");

  // LSP's SymbolKind doesn't have a "Macro" kind, and
  // indexSymbolKindToSymbolKind() currently maps macros
  // to SymbolKind::String.
  EXPECT_THAT(getSymbols("macro"),
              ElementsAre(AllOf(QName("MACRO"), WithKind(SymbolKind::String))));
}

TEST_F(WorkspaceSymbolsTest, NoLocals) {
  addFile("foo.cpp", R"cpp(
      void test(int FirstParam, int SecondParam) {
        struct LocalClass {};
        int local_var;
      })cpp");
  EXPECT_THAT(getSymbols("l"), IsEmpty());
  EXPECT_THAT(getSymbols("p"), IsEmpty());
}

TEST_F(WorkspaceSymbolsTest, Globals) {
  addFile("foo.h", R"cpp(
      int global_var;

      int global_func();

      struct GlobalStruct {};)cpp");
  addFile("foo.cpp", R"cpp(
      #include "foo.h"
      )cpp");
  EXPECT_THAT(getSymbols("global"),
              UnorderedElementsAre(
                  AllOf(QName("GlobalStruct"), WithKind(SymbolKind::Struct)),
                  AllOf(QName("global_func"), WithKind(SymbolKind::Function)),
                  AllOf(QName("global_var"), WithKind(SymbolKind::Variable))));
}

TEST_F(WorkspaceSymbolsTest, Unnamed) {
  addFile("foo.h", R"cpp(
      struct {
        int InUnnamed;
      } UnnamedStruct;)cpp");
  addFile("foo.cpp", R"cpp(
      #include "foo.h"
      )cpp");
  EXPECT_THAT(getSymbols("UnnamedStruct"),
              ElementsAre(AllOf(QName("UnnamedStruct"),
                                WithKind(SymbolKind::Variable))));
  EXPECT_THAT(getSymbols("InUnnamed"),
              ElementsAre(AllOf(QName("(anonymous struct)::InUnnamed"),
                                WithKind(SymbolKind::Field))));
}

TEST_F(WorkspaceSymbolsTest, InMainFile) {
  addFile("foo.cpp", R"cpp(
      int test() {}
      static test2() {}
      )cpp");
  EXPECT_THAT(getSymbols("test"), ElementsAre(QName("test"), QName("test2")));
}

TEST_F(WorkspaceSymbolsTest, Namespaces) {
  addFile("foo.h", R"cpp(
      namespace ans1 {
        int ai1;
      namespace ans2 {
        int ai2;
      }
      }
      )cpp");
  addFile("foo.cpp", R"cpp(
      #include "foo.h"
      )cpp");
  EXPECT_THAT(getSymbols("a"),
              UnorderedElementsAre(QName("ans1"), QName("ans1::ai1"),
                                   QName("ans1::ans2"),
                                   QName("ans1::ans2::ai2")));
  EXPECT_THAT(getSymbols("::"), ElementsAre(QName("ans1")));
  EXPECT_THAT(getSymbols("::a"), ElementsAre(QName("ans1")));
  EXPECT_THAT(getSymbols("ans1::"),
              UnorderedElementsAre(QName("ans1::ai1"), QName("ans1::ans2")));
  EXPECT_THAT(getSymbols("::ans1"), ElementsAre(QName("ans1")));
  EXPECT_THAT(getSymbols("::ans1::"),
              UnorderedElementsAre(QName("ans1::ai1"), QName("ans1::ans2")));
  EXPECT_THAT(getSymbols("::ans1::ans2"), ElementsAre(QName("ans1::ans2")));
  EXPECT_THAT(getSymbols("::ans1::ans2::"),
              ElementsAre(QName("ans1::ans2::ai2")));
}

TEST_F(WorkspaceSymbolsTest, AnonymousNamespace) {
  addFile("foo.cpp", R"cpp(
      namespace {
      void test() {}
      }
      )cpp");
  EXPECT_THAT(getSymbols("test"), ElementsAre(QName("test")));
}

TEST_F(WorkspaceSymbolsTest, MultiFile) {
  addFile("foo.h", R"cpp(
      int foo() {
      }
      )cpp");
  addFile("foo2.h", R"cpp(
      int foo2() {
      }
      )cpp");
  addFile("foo.cpp", R"cpp(
      #include "foo.h"
      #include "foo2.h"
      )cpp");
  EXPECT_THAT(getSymbols("foo"),
              UnorderedElementsAre(QName("foo"), QName("foo2")));
}

TEST_F(WorkspaceSymbolsTest, GlobalNamespaceQueries) {
  addFile("foo.h", R"cpp(
      int foo() {
      }
      class Foo {
        int a;
      };
      namespace ns {
      int foo2() {
      }
      }
      )cpp");
  addFile("foo.cpp", R"cpp(
      #include "foo.h"
      )cpp");
  EXPECT_THAT(getSymbols("::"),
              UnorderedElementsAre(
                  AllOf(QName("Foo"), WithKind(SymbolKind::Class)),
                  AllOf(QName("foo"), WithKind(SymbolKind::Function)),
                  AllOf(QName("ns"), WithKind(SymbolKind::Namespace))));
  EXPECT_THAT(getSymbols(":"), IsEmpty());
  EXPECT_THAT(getSymbols(""), IsEmpty());
}

TEST_F(WorkspaceSymbolsTest, Enums) {
  addFile("foo.h", R"cpp(
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
      )cpp");
  addFile("foo.cpp", R"cpp(
      #include "foo.h"
      )cpp");
  EXPECT_THAT(getSymbols("Red"), ElementsAre(QName("Red")));
  EXPECT_THAT(getSymbols("::Red"), ElementsAre(QName("Red")));
  EXPECT_THAT(getSymbols("Green"), ElementsAre(QName("Green")));
  EXPECT_THAT(getSymbols("Green"), ElementsAre(QName("Green")));
  EXPECT_THAT(getSymbols("Color2::Yellow"),
              ElementsAre(QName("Color2::Yellow")));
  EXPECT_THAT(getSymbols("Yellow"), ElementsAre(QName("Color2::Yellow")));

  EXPECT_THAT(getSymbols("ns::Black"), ElementsAre(QName("ns::Black")));
  EXPECT_THAT(getSymbols("ns::Blue"), ElementsAre(QName("ns::Blue")));
  EXPECT_THAT(getSymbols("ns::Color4::White"),
              ElementsAre(QName("ns::Color4::White")));
}

TEST_F(WorkspaceSymbolsTest, Ranking) {
  addFile("foo.h", R"cpp(
      namespace ns{}
      void func();
      )cpp");
  addFile("foo.cpp", R"cpp(
      #include "foo.h"
      )cpp");
  EXPECT_THAT(getSymbols("::"), ElementsAre(QName("func"), QName("ns")));
}

TEST_F(WorkspaceSymbolsTest, WithLimit) {
  addFile("foo.h", R"cpp(
      int foo;
      int foo2;
      )cpp");
  addFile("foo.cpp", R"cpp(
      #include "foo.h"
      )cpp");
  // Foo is higher ranked because of exact name match.
  EXPECT_THAT(getSymbols("foo"),
              UnorderedElementsAre(
                  AllOf(QName("foo"), WithKind(SymbolKind::Variable)),
                  AllOf(QName("foo2"), WithKind(SymbolKind::Variable))));

  Limit = 1;
  EXPECT_THAT(getSymbols("foo"), ElementsAre(QName("foo")));
}

TEST_F(WorkspaceSymbolsTest, TempSpecs) {
  addFile("foo.h", R"cpp(
      template <typename T, typename U, int X = 5> class Foo {};
      template <typename T> class Foo<int, T> {};
      template <> class Foo<bool, int> {};
      template <> class Foo<bool, int, 3> {};
      )cpp");
  // Foo is higher ranked because of exact name match.
  EXPECT_THAT(
      getSymbols("Foo"),
      UnorderedElementsAre(
          AllOf(QName("Foo"), WithKind(SymbolKind::Class)),
          AllOf(QName("Foo<int, T>"), WithKind(SymbolKind::Class)),
          AllOf(QName("Foo<bool, int>"), WithKind(SymbolKind::Class)),
          AllOf(QName("Foo<bool, int, 3>"), WithKind(SymbolKind::Class))));
}

namespace {
class DocumentSymbolsTest : public ::testing::Test {
public:
  DocumentSymbolsTest() : Server(CDB, FSProvider, optsForTests()) {}

protected:
  MockFSProvider FSProvider;
  MockCompilationDatabase CDB;
  ClangdServer Server;

  std::vector<DocumentSymbol> getSymbols(PathRef File) {
    EXPECT_TRUE(Server.blockUntilIdleForTest()) << "Waiting for preamble";
    auto SymbolInfos = runDocumentSymbols(Server, File);
    EXPECT_TRUE(bool(SymbolInfos)) << "documentSymbols returned an error";
    return *SymbolInfos;
  }

  void addFile(llvm::StringRef FilePath, llvm::StringRef Contents) {
    FSProvider.Files[FilePath] = std::string(Contents);
    Server.addDocument(FilePath, Contents);
  }
};
} // namespace

TEST_F(DocumentSymbolsTest, BasicSymbols) {
  std::string FilePath = testPath("foo.cpp");
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

  addFile(FilePath, Main.code());
  EXPECT_THAT(
      getSymbols(FilePath),
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

TEST_F(DocumentSymbolsTest, DeclarationDefinition) {
  std::string FilePath = testPath("foo.cpp");
  Annotations Main(R"(
      class Foo {
        void $decl[[f]]();
      };
      void Foo::$def[[f]]() {
      }
    )");

  addFile(FilePath, Main.code());
  EXPECT_THAT(
      getSymbols(FilePath),
      ElementsAre(
          AllOf(WithName("Foo"), WithKind(SymbolKind::Class),
                Children(AllOf(WithName("f"), WithKind(SymbolKind::Method),
                               SymNameRange(Main.range("decl"))))),
          AllOf(WithName("Foo::f"), WithKind(SymbolKind::Method),
                SymNameRange(Main.range("def")))));
}

TEST_F(DocumentSymbolsTest, Concepts) {
  CDB.ExtraClangFlags = {"-std=c++20"};
  std::string FilePath = testPath("foo.cpp");
  addFile(FilePath,
          "template <typename T> concept C = requires(T t) { t.foo(); };");

  EXPECT_THAT(getSymbols(FilePath), ElementsAre(WithName("C")));
}

TEST_F(DocumentSymbolsTest, ExternSymbol) {
  std::string FilePath = testPath("foo.cpp");
  addFile(testPath("foo.h"), R"cpp(
      extern int var;
      )cpp");
  addFile(FilePath, R"cpp(
      #include "foo.h"
      )cpp");

  EXPECT_THAT(getSymbols(FilePath), IsEmpty());
}

TEST_F(DocumentSymbolsTest, NoLocals) {
  std::string FilePath = testPath("foo.cpp");
  addFile(FilePath,
          R"cpp(
      void test(int FirstParam, int SecondParam) {
        struct LocalClass {};
        int local_var;
      })cpp");
  EXPECT_THAT(getSymbols(FilePath), ElementsAre(WithName("test")));
}

TEST_F(DocumentSymbolsTest, Unnamed) {
  std::string FilePath = testPath("foo.h");
  addFile(FilePath,
          R"cpp(
      struct {
        int InUnnamed;
      } UnnamedStruct;
      )cpp");
  EXPECT_THAT(
      getSymbols(FilePath),
      ElementsAre(
          AllOf(WithName("(anonymous struct)"), WithKind(SymbolKind::Struct),
                Children(AllOf(WithName("InUnnamed"),
                               WithKind(SymbolKind::Field), Children()))),
          AllOf(WithName("UnnamedStruct"), WithKind(SymbolKind::Variable),
                Children())));
}

TEST_F(DocumentSymbolsTest, InHeaderFile) {
  addFile(testPath("bar.h"), R"cpp(
      int foo() {
      }
      )cpp");
  std::string FilePath = testPath("foo.h");
  addFile(FilePath, R"cpp(
      #include "bar.h"
      int test() {
      }
      )cpp");
  addFile(testPath("foo.cpp"), R"cpp(
      #include "foo.h"
      )cpp");
  EXPECT_THAT(getSymbols(FilePath), ElementsAre(WithName("test")));
}

TEST_F(DocumentSymbolsTest, Template) {
  std::string FilePath = testPath("foo.cpp");
  addFile(FilePath, R"(
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
  )");
  EXPECT_THAT(
      getSymbols(FilePath),
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

TEST_F(DocumentSymbolsTest, Namespaces) {
  std::string FilePath = testPath("foo.cpp");
  addFile(FilePath, R"cpp(
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
      )cpp");
  EXPECT_THAT(
      getSymbols(FilePath),
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

TEST_F(DocumentSymbolsTest, Enums) {
  std::string FilePath = testPath("foo.cpp");
  addFile(FilePath, R"(
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
    )");
  EXPECT_THAT(
      getSymbols(FilePath),
      ElementsAre(
          AllOf(WithName("(anonymous enum)"), Children(WithName("Red"))),
          AllOf(WithName("Color"), Children(WithName("Green"))),
          AllOf(WithName("Color2"), Children(WithName("Yellow"))),
          AllOf(WithName("ns"), Children(AllOf(WithName("(anonymous enum)"),
                                               Children(WithName("Black")))))));
}

TEST_F(DocumentSymbolsTest, FromMacro) {
  std::string FilePath = testPath("foo.cpp");
  Annotations Main(R"(
    #define FF(name) \
      class name##_Test {};

    $expansion[[FF]](abc);

    #define FF2() \
      class $spelling[[Test]] {};

    FF2();
  )");
  addFile(FilePath, Main.code());
  EXPECT_THAT(
      getSymbols(FilePath),
      ElementsAre(
          AllOf(WithName("abc_Test"), SymNameRange(Main.range("expansion"))),
          AllOf(WithName("Test"), SymNameRange(Main.range("spelling")))));
}

TEST_F(DocumentSymbolsTest, FuncTemplates) {
  std::string FilePath = testPath("foo.cpp");
  Annotations Source(R"cpp(
    template <class T>
    T foo() {}

    auto x = foo<int>();
    auto y = foo<double>()
  )cpp");
  addFile(FilePath, Source.code());
  // Make sure we only see the template declaration, not instantiations.
  EXPECT_THAT(getSymbols(FilePath),
              ElementsAre(WithName("foo"), WithName("x"), WithName("y")));
}

TEST_F(DocumentSymbolsTest, UsingDirectives) {
  std::string FilePath = testPath("foo.cpp");
  Annotations Source(R"cpp(
    namespace ns {
      int foo;
    }

    namespace ns_alias = ns;

    using namespace ::ns;     // check we don't loose qualifiers.
    using namespace ns_alias; // and namespace aliases.
  )cpp");
  addFile(FilePath, Source.code());
  EXPECT_THAT(getSymbols(FilePath),
              ElementsAre(WithName("ns"), WithName("ns_alias"),
                          WithName("using namespace ::ns"),
                          WithName("using namespace ns_alias")));
}

TEST_F(DocumentSymbolsTest, TempSpecs) {
  std::string FilePath = testPath("foo.cpp");
  addFile(FilePath, R"cpp(
      template <typename T, typename U, int X = 5> class Foo {};
      template <typename T> class Foo<int, T> {};
      template <> class Foo<bool, int> {};
      template <> class Foo<bool, int, 3> {};
      )cpp");
  // Foo is higher ranked because of exact name match.
  EXPECT_THAT(
      getSymbols(FilePath),
      UnorderedElementsAre(
          AllOf(WithName("Foo"), WithKind(SymbolKind::Class)),
          AllOf(WithName("Foo<int, T>"), WithKind(SymbolKind::Class)),
          AllOf(WithName("Foo<bool, int>"), WithKind(SymbolKind::Class)),
          AllOf(WithName("Foo<bool, int, 3>"), WithKind(SymbolKind::Class))));
}

TEST_F(DocumentSymbolsTest, Qualifiers) {
  std::string FilePath = testPath("foo.cpp");
  addFile(FilePath, R"cpp(
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
  )cpp");

  // All the qualifiers should be preserved exactly as written.
  EXPECT_THAT(getSymbols(FilePath),
              UnorderedElementsAre(
                  WithName("foo"), WithName("foo::bar::Cls"),
                  WithName("foo::bar::func1"), WithName("::foo::bar::func2"),
                  WithName("using namespace foo"), WithName("bar::func3"),
                  WithName("alias"), WithName("::alias::func4")));
}

TEST_F(DocumentSymbolsTest, QualifiersWithTemplateArgs) {
  std::string FilePath = testPath("foo.cpp");
  addFile(FilePath, R"cpp(
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
      )cpp");
  EXPECT_THAT(
      getSymbols(FilePath),
      UnorderedElementsAre(WithName("Foo"), WithName("Foo<int, double>"),
                           WithName("int_type"),
                           WithName("Foo<int_type, double>::method1"),
                           WithName("Foo<int>::method2"), WithName("Foo_type"),
                           WithName("Foo_type::method3")));
}

} // namespace clangd
} // namespace clang
