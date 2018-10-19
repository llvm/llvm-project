//===-- FindSymbolsTests.cpp -------------------------*- C++ -*------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
using ::testing::AnyOf;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

class IgnoreDiagnostics : public DiagnosticsConsumer {
  void onDiagnosticsReady(PathRef File,
                          std::vector<Diag> Diagnostics) override {}
};

// GMock helpers for matching SymbolInfos items.
MATCHER_P(QName, Name, "") {
  if (arg.containerName.empty())
    return arg.name == Name;
  return (arg.containerName + "::" + arg.name) == Name;
}
MATCHER_P(WithKind, Kind, "") { return arg.kind == Kind; }
MATCHER_P(SymRange, Range, "") { return arg.location.range == Range; }

ClangdServer::Options optsForTests() {
  auto ServerOpts = ClangdServer::optsForTest();
  ServerOpts.WorkspaceRoot = testRoot();
  ServerOpts.BuildDynamicSymbolIndex = true;
  ServerOpts.URISchemes = {"unittest", "file"};
  return ServerOpts;
}

class WorkspaceSymbolsTest : public ::testing::Test {
public:
  WorkspaceSymbolsTest()
      : Server(CDB, FSProvider, DiagConsumer, optsForTests()) {
    // Make sure the test root directory is created.
    FSProvider.Files[testPath("unused")] = "";
  }

protected:
  MockFSProvider FSProvider;
  MockCompilationDatabase CDB;
  IgnoreDiagnostics DiagConsumer;
  ClangdServer Server;
  int Limit = 0;

  std::vector<SymbolInformation> getSymbols(StringRef Query) {
    EXPECT_TRUE(Server.blockUntilIdleForTest()) << "Waiting for preamble";
    auto SymbolInfos = runWorkspaceSymbols(Server, Query, Limit);
    EXPECT_TRUE(bool(SymbolInfos)) << "workspaceSymbols returned an error";
    return *SymbolInfos;
  }

  void addFile(StringRef FileName, StringRef Contents) {
    auto Path = testPath(FileName);
    FSProvider.Files[Path] = Contents;
    Server.addDocument(Path, Contents);
  }
};

} // namespace

TEST_F(WorkspaceSymbolsTest, NoMacro) {
  addFile("foo.cpp", R"cpp(
      #define MACRO X
      )cpp");

  // Macros are not in the index.
  EXPECT_THAT(getSymbols("macro"), IsEmpty());
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
      int test() {
      }
      )cpp");
  EXPECT_THAT(getSymbols("test"), IsEmpty());
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
  addFile("foo.h", R"cpp(
      namespace {
      void test() {}
      }
      )cpp");
  addFile("foo.cpp", R"cpp(
      #include "foo.h"
      )cpp");
  EXPECT_THAT(getSymbols("test"), IsEmpty());
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
      function func();
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

namespace {
class DocumentSymbolsTest : public ::testing::Test {
public:
  DocumentSymbolsTest()
      : Server(CDB, FSProvider, DiagConsumer, optsForTests()) {}

protected:
  MockFSProvider FSProvider;
  MockCompilationDatabase CDB;
  IgnoreDiagnostics DiagConsumer;
  ClangdServer Server;

  std::vector<SymbolInformation> getSymbols(PathRef File) {
    EXPECT_TRUE(Server.blockUntilIdleForTest()) << "Waiting for preamble";
    auto SymbolInfos = runDocumentSymbols(Server, File);
    EXPECT_TRUE(bool(SymbolInfos)) << "documentSymbols returned an error";
    return *SymbolInfos;
  }

  void addFile(StringRef FilePath, StringRef Contents) {
    FSProvider.Files[FilePath] = Contents;
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

      // FIXME: using declaration is not supported as the IndexAction will ignore
      // implicit declarations (the implicit using shadow declaration) by default,
      // and there is no way to customize this behavior at the moment.
      using bar::v2;
      } // namespace foo
    )");

  addFile(FilePath, Main.code());
  EXPECT_THAT(getSymbols(FilePath),
              ElementsAreArray(
                  {AllOf(QName("Foo"), WithKind(SymbolKind::Class)),
                   AllOf(QName("Foo"), WithKind(SymbolKind::Class)),
                   AllOf(QName("Foo::Foo"), WithKind(SymbolKind::Method)),
                   AllOf(QName("Foo::Foo"), WithKind(SymbolKind::Method)),
                   AllOf(QName("Foo::f"), WithKind(SymbolKind::Method)),
                   AllOf(QName("f1"), WithKind(SymbolKind::Function)),
                   AllOf(QName("Foo::operator="), WithKind(SymbolKind::Method)),
                   AllOf(QName("Foo::~Foo"), WithKind(SymbolKind::Method)),
                   AllOf(QName("Foo::Nested"), WithKind(SymbolKind::Class)),
                   AllOf(QName("Foo::Nested::f"), WithKind(SymbolKind::Method)),
                   AllOf(QName("Friend"), WithKind(SymbolKind::Class)),
                   AllOf(QName("f1"), WithKind(SymbolKind::Function)),
                   AllOf(QName("f2"), WithKind(SymbolKind::Function)),
                   AllOf(QName("KInt"), WithKind(SymbolKind::Variable)),
                   AllOf(QName("kStr"), WithKind(SymbolKind::Variable)),
                   AllOf(QName("f1"), WithKind(SymbolKind::Function)),
                   AllOf(QName("foo"), WithKind(SymbolKind::Namespace)),
                   AllOf(QName("foo::int32"), WithKind(SymbolKind::Class)),
                   AllOf(QName("foo::int32_t"), WithKind(SymbolKind::Class)),
                   AllOf(QName("foo::v1"), WithKind(SymbolKind::Variable)),
                   AllOf(QName("foo::bar"), WithKind(SymbolKind::Namespace)),
                   AllOf(QName("foo::bar::v2"), WithKind(SymbolKind::Variable)),
                   AllOf(QName("foo::baz"), WithKind(SymbolKind::Namespace))}));
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
  EXPECT_THAT(getSymbols(FilePath),
              ElementsAre(AllOf(QName("Foo"), WithKind(SymbolKind::Class)),
                          AllOf(QName("Foo::f"), WithKind(SymbolKind::Method),
                                SymRange(Main.range("decl"))),
                          AllOf(QName("Foo::f"), WithKind(SymbolKind::Method),
                                SymRange(Main.range("def")))));
}

TEST_F(DocumentSymbolsTest, ExternSymbol) {
  std::string FilePath = testPath("foo.cpp");
  addFile(testPath("foo.h"), R"cpp(
      extern int var = 2;
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
  EXPECT_THAT(getSymbols(FilePath), ElementsAre(QName("test")));
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
      ElementsAre(AllOf(QName("UnnamedStruct"), WithKind(SymbolKind::Variable)),
                  AllOf(QName("(anonymous struct)::InUnnamed"),
                        WithKind(SymbolKind::Field))));
}

TEST_F(DocumentSymbolsTest, InHeaderFile) {
  addFile("bar.h", R"cpp(
      int foo() {
      }
      )cpp");
  std::string FilePath = testPath("foo.h");
  addFile(FilePath, R"cpp(
      #include "bar.h"
      int test() {
      }
      )cpp");
  addFile("foo.cpp", R"cpp(
      #include "foo.h"
      )cpp");
  EXPECT_THAT(getSymbols(FilePath), ElementsAre(QName("test")));
}

TEST_F(DocumentSymbolsTest, Template) {
  std::string FilePath = testPath("foo.cpp");
  addFile(FilePath, R"(
    // Primary templates and specializations are included but instantiations
    // are not.
    template <class T> struct Tmpl {T x = 0;};
    template <> struct Tmpl<int> {};
    extern template struct Tmpl<float>;
    template struct Tmpl<double>;
  )");
  EXPECT_THAT(getSymbols(FilePath),
              ElementsAre(AllOf(QName("Tmpl"), WithKind(SymbolKind::Struct)),
                          AllOf(QName("Tmpl::x"), WithKind(SymbolKind::Field)),
                          AllOf(QName("Tmpl"), WithKind(SymbolKind::Struct))));
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
      ElementsAreArray({QName("ans1"), QName("ans1::ai1"), QName("ans1::ans2"),
                        QName("ans1::ans2::ai2"), QName("test"), QName("na"),
                        QName("na::nb"), QName("na::Foo"), QName("na"),
                        QName("na::nb"), QName("na::Bar")}));
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
  EXPECT_THAT(getSymbols(FilePath),
              ElementsAre(QName("Red"), QName("Color"), QName("Green"),
                          QName("Color2"), QName("Color2::Yellow"), QName("ns"),
                          QName("ns::Black")));
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
      ElementsAre(AllOf(QName("abc_Test"), SymRange(Main.range("expansion"))),
                  AllOf(QName("Test"), SymRange(Main.range("spelling")))));
}

} // namespace clangd
} // namespace clang
