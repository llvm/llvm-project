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
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

class IgnoreDiagnostics : public DiagnosticsConsumer {
  void onDiagnosticsReady(PathRef File,
                          std::vector<Diag> Diagnostics) override {}
};

// GMock helpers for matching SymbolInfos items.
MATCHER_P(Named, Name, "") { return arg.name == Name; }
MATCHER_P(InContainer, ContainerName, "") {
  return arg.containerName == ContainerName;
}
MATCHER_P(WithKind, Kind, "") { return arg.kind == Kind; }

ClangdServer::Options optsForTests() {
  auto ServerOpts = ClangdServer::optsForTest();
  ServerOpts.BuildDynamicSymbolIndex = true;
  return ServerOpts;
}

class WorkspaceSymbolsTest : public ::testing::Test {
public:
  WorkspaceSymbolsTest()
      : Server(CDB, FSProvider, DiagConsumer, optsForTests()) {}

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
              UnorderedElementsAre(AllOf(Named("GlobalStruct"), InContainer(""),
                                         WithKind(SymbolKind::Struct)),
                                   AllOf(Named("global_func"), InContainer(""),
                                         WithKind(SymbolKind::Function)),
                                   AllOf(Named("global_var"), InContainer(""),
                                         WithKind(SymbolKind::Variable))));
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
              ElementsAre(AllOf(Named("UnnamedStruct"),
                                WithKind(SymbolKind::Variable))));
  EXPECT_THAT(
      getSymbols("InUnnamed"),
      ElementsAre(AllOf(Named("InUnnamed"), InContainer("(anonymous struct)"),
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
  EXPECT_THAT(
      getSymbols("a"),
      UnorderedElementsAre(AllOf(Named("ans1"), InContainer("")),
                           AllOf(Named("ai1"), InContainer("ans1")),
                           AllOf(Named("ans2"), InContainer("ans1")),
                           AllOf(Named("ai2"), InContainer("ans1::ans2"))));
  EXPECT_THAT(getSymbols("::"),
              ElementsAre(AllOf(Named("ans1"), InContainer(""))));
  EXPECT_THAT(getSymbols("::a"),
              ElementsAre(AllOf(Named("ans1"), InContainer(""))));
  EXPECT_THAT(getSymbols("ans1::"),
              UnorderedElementsAre(AllOf(Named("ai1"), InContainer("ans1")),
                                   AllOf(Named("ans2"), InContainer("ans1"))));
  EXPECT_THAT(getSymbols("::ans1"),
              ElementsAre(AllOf(Named("ans1"), InContainer(""))));
  EXPECT_THAT(getSymbols("::ans1::"),
              UnorderedElementsAre(AllOf(Named("ai1"), InContainer("ans1")),
                                   AllOf(Named("ans2"), InContainer("ans1"))));
  EXPECT_THAT(getSymbols("::ans1::ans2"),
              ElementsAre(AllOf(Named("ans2"), InContainer("ans1"))));
  EXPECT_THAT(getSymbols("::ans1::ans2::"),
              ElementsAre(AllOf(Named("ai2"), InContainer("ans1::ans2"))));
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
              UnorderedElementsAre(AllOf(Named("foo"), InContainer("")),
                                   AllOf(Named("foo2"), InContainer(""))));
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
  EXPECT_THAT(
      getSymbols("::"),
      UnorderedElementsAre(
          AllOf(Named("Foo"), InContainer(""), WithKind(SymbolKind::Class)),
          AllOf(Named("foo"), InContainer(""), WithKind(SymbolKind::Function)),
          AllOf(Named("ns"), InContainer(""),
                WithKind(SymbolKind::Namespace))));
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
  EXPECT_THAT(getSymbols("Red"), ElementsAre(Named("Red")));
  EXPECT_THAT(getSymbols("::Red"), ElementsAre(Named("Red")));
  EXPECT_THAT(getSymbols("Green"), ElementsAre(Named("Green")));
  EXPECT_THAT(getSymbols("Green"), ElementsAre(Named("Green")));
  EXPECT_THAT(getSymbols("Color2::Yellow"), ElementsAre(Named("Yellow")));
  EXPECT_THAT(getSymbols("Yellow"), ElementsAre(Named("Yellow")));

  EXPECT_THAT(getSymbols("ns::Black"), ElementsAre(Named("Black")));
  EXPECT_THAT(getSymbols("ns::Blue"), ElementsAre(Named("Blue")));
  EXPECT_THAT(getSymbols("ns::Color4::White"), ElementsAre(Named("White")));
}

TEST_F(WorkspaceSymbolsTest, Ranking) {
  addFile("foo.h", R"cpp(
      namespace ns{}
      function func();
      )cpp");
  addFile("foo.cpp", R"cpp(
      #include "foo.h"
      )cpp");
  EXPECT_THAT(getSymbols("::"), ElementsAre(Named("func"), Named("ns")));
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
              UnorderedElementsAre(AllOf(Named("foo"), InContainer(""),
                                         WithKind(SymbolKind::Variable)),
                                   AllOf(Named("foo2"), InContainer(""),
                                         WithKind(SymbolKind::Variable))));

  Limit = 1;
  EXPECT_THAT(getSymbols("foo"), ElementsAre(Named("foo")));
}

} // namespace clangd
} // namespace clang
