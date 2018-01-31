//===-- SymbolCollectorTests.cpp  -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "TestFS.h"
#include "index/SymbolCollector.h"
#include "index/SymbolYAML.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/VirtualFileSystem.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Index/IndexingAction.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <memory>
#include <string>

using testing::AllOf;
using testing::Eq;
using testing::Field;
using testing::Not;
using testing::UnorderedElementsAre;
using testing::UnorderedElementsAreArray;

// GMock helpers for matching Symbol.
MATCHER_P(Labeled, Label, "") { return arg.CompletionLabel == Label; }
MATCHER(HasDetail, "") { return arg.Detail; }
MATCHER_P(Detail, D, "") {
  return arg.Detail && arg.Detail->CompletionDetail == D;
}
MATCHER_P(Doc, D, "") { return arg.Detail && arg.Detail->Documentation == D; }
MATCHER_P(Plain, Text, "") { return arg.CompletionPlainInsertText == Text; }
MATCHER_P(Snippet, S, "") {
  return arg.CompletionSnippetInsertText == S;
}
MATCHER_P(QName, Name, "") { return (arg.Scope + arg.Name).str() == Name; }
MATCHER_P(CPath, P, "") { return arg.CanonicalDeclaration.FilePath == P; }
MATCHER_P(LocationOffsets, Offsets, "") {
  // Offset range in SymbolLocation is [start, end] while in Clangd is [start,
  // end).
  return arg.CanonicalDeclaration.StartOffset == Offsets.first &&
      arg.CanonicalDeclaration.EndOffset == Offsets.second - 1;
}
//MATCHER_P(FilePath, P, "") {
  //return arg.CanonicalDeclaration.FilePath.contains(P);
//}

namespace clang {
namespace clangd {

namespace {
const char TestHeaderName[] = "symbols.h";
const char TestFileName[] = "symbol.cc";
class SymbolIndexActionFactory : public tooling::FrontendActionFactory {
public:
  SymbolIndexActionFactory(SymbolCollector::Options COpts)
      : COpts(std::move(COpts)) {}

  clang::FrontendAction *create() override {
    index::IndexingOptions IndexOpts;
    IndexOpts.SystemSymbolFilter =
        index::IndexingOptions::SystemSymbolFilterKind::All;
    IndexOpts.IndexFunctionLocals = false;
    Collector = std::make_shared<SymbolCollector>(COpts);
    FrontendAction *Action =
        index::createIndexingAction(Collector, IndexOpts, nullptr).release();
    return Action;
  }

  std::shared_ptr<SymbolCollector> Collector;
  SymbolCollector::Options COpts;
};

class SymbolCollectorTest : public ::testing::Test {
public:
  bool runSymbolCollector(StringRef HeaderCode, StringRef MainCode) {
    llvm::IntrusiveRefCntPtr<vfs::InMemoryFileSystem> InMemoryFileSystem(
        new vfs::InMemoryFileSystem);
    llvm::IntrusiveRefCntPtr<FileManager> Files(
        new FileManager(FileSystemOptions(), InMemoryFileSystem));

    auto Factory = llvm::make_unique<SymbolIndexActionFactory>(CollectorOpts);

    tooling::ToolInvocation Invocation(
        {"symbol_collector", "-fsyntax-only", "-std=c++11", TestFileName},
        Factory->create(), Files.get(),
        std::make_shared<PCHContainerOperations>());

    InMemoryFileSystem->addFile(TestHeaderName, 0,
                                llvm::MemoryBuffer::getMemBuffer(HeaderCode));

    std::string Content = MainCode;
    if (!HeaderCode.empty())
      Content = "#include\"" + std::string(TestHeaderName) + "\"\n" + Content;
    InMemoryFileSystem->addFile(TestFileName, 0,
                                llvm::MemoryBuffer::getMemBuffer(Content));
    Invocation.run();
    Symbols = Factory->Collector->takeSymbols();
    return true;
  }

protected:
  SymbolSlab Symbols;
  SymbolCollector::Options CollectorOpts;
};

TEST_F(SymbolCollectorTest, CollectSymbols) {
  CollectorOpts.IndexMainFiles = true;
  const std::string Header = R"(
    class Foo {
      void f();
    };
    void f1();
    inline void f2() {}
    static const int KInt = 2;
    const char* kStr = "123";
  )";
  const std::string Main = R"(
    namespace {
    void ff() {} // ignore
    }

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
  )";
  runSymbolCollector(Header, Main);
  EXPECT_THAT(Symbols,
              UnorderedElementsAreArray(
                  {QName("Foo"), QName("f1"), QName("f2"), QName("KInt"),
                   QName("kStr"), QName("foo"), QName("foo::bar"),
                   QName("foo::int32"), QName("foo::int32_t"), QName("foo::v1"),
                   QName("foo::bar::v2"), QName("foo::baz")}));
}

TEST_F(SymbolCollectorTest, SymbolRelativeNoFallback) {
  CollectorOpts.IndexMainFiles = false;
  runSymbolCollector("class Foo {};", /*Main=*/"");
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(AllOf(QName("Foo"), CPath("symbols.h"))));
}

TEST_F(SymbolCollectorTest, SymbolRelativeWithFallback) {
  CollectorOpts.IndexMainFiles = false;
  CollectorOpts.FallbackDir = getVirtualTestRoot();
  runSymbolCollector("class Foo {};", /*Main=*/"");
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(AllOf(
                  QName("Foo"), CPath(getVirtualTestFilePath("symbols.h")))));
}

TEST_F(SymbolCollectorTest, IncludeEnums) {
  CollectorOpts.IndexMainFiles = false;
  const std::string Header = R"(
    enum {
      Red
    };
    enum Color {
      Green
    };
    enum class Color2 {
      Yellow // ignore
    };
    namespace ns {
    enum {
      Black
    };
    }
  )";
  runSymbolCollector(Header, /*Main=*/"");
  EXPECT_THAT(Symbols, UnorderedElementsAre(QName("Red"), QName("Color"),
                                            QName("Green"), QName("Color2"),
                                            QName("ns"),
                                            QName("ns::Black")));
}

TEST_F(SymbolCollectorTest, IgnoreNamelessSymbols) {
  CollectorOpts.IndexMainFiles = false;
  const std::string Header = R"(
    struct {
      int a;
    } Foo;
  )";
  runSymbolCollector(Header, /*Main=*/"");
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(QName("Foo")));
}

TEST_F(SymbolCollectorTest, SymbolFormedFromMacro) {
  CollectorOpts.IndexMainFiles = false;

  Annotations Header(R"(
    #define FF(name) \
      class name##_Test {};

    $expansion[[FF(abc)]];

    #define FF2() \
      $spelling[[class Test {}]];

    FF2();
  )");

  runSymbolCollector(Header.code(), /*Main=*/"");
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(
                  AllOf(QName("abc_Test"),
                        LocationOffsets(Header.offsetRange("expansion")),
                        CPath(TestHeaderName)),
                  AllOf(QName("Test"),
                        LocationOffsets(Header.offsetRange("spelling")),
                        CPath(TestHeaderName))));
}

TEST_F(SymbolCollectorTest, SymbolFormedFromMacroInMainFile) {
  CollectorOpts.IndexMainFiles = true;

  Annotations Main(R"(
    #define FF(name) \
      class name##_Test {};

    $expansion[[FF(abc)]];

    #define FF2() \
      $spelling[[class Test {}]];

    FF2();
  )");
  runSymbolCollector(/*Header=*/"", Main.code());
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(
                  AllOf(QName("abc_Test"),
                        LocationOffsets(Main.offsetRange("expansion")),
                        CPath(TestFileName)),
                  AllOf(QName("Test"),
                        LocationOffsets(Main.offsetRange("spelling")),
                        CPath(TestFileName))));
}

TEST_F(SymbolCollectorTest, IgnoreSymbolsInMainFile) {
  CollectorOpts.IndexMainFiles = false;
  const std::string Header = R"(
    class Foo {};
    void f1();
    inline void f2() {}
  )";
  const std::string Main = R"(
    namespace {
    void ff() {} // ignore
    }
    void main_f() {} // ignore
    void f1() {}
  )";
  runSymbolCollector(Header, Main);
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(QName("Foo"), QName("f1"), QName("f2")));
}

TEST_F(SymbolCollectorTest, IncludeSymbolsInMainFile) {
  CollectorOpts.IndexMainFiles = true;
  const std::string Header = R"(
    class Foo {};
    void f1();
    inline void f2() {}
  )";
  const std::string Main = R"(
    namespace {
    void ff() {} // ignore
    }
    void main_f() {}
    void f1() {}
  )";
  runSymbolCollector(Header, Main);
  EXPECT_THAT(Symbols, UnorderedElementsAre(QName("Foo"), QName("f1"),
                                            QName("f2"), QName("main_f")));
}

TEST_F(SymbolCollectorTest, IgnoreClassMembers) {
  const std::string Header = R"(
    class Foo {
      void f() {}
      void g();
      static void sf() {}
      static void ssf();
      static int x;
    };
  )";
  const std::string Main = R"(
    void Foo::g() {}
    void Foo::ssf() {}
  )";
  runSymbolCollector(Header, Main);
  EXPECT_THAT(Symbols, UnorderedElementsAre(QName("Foo")));
}

TEST_F(SymbolCollectorTest, SymbolWithDocumentation) {
  const std::string Header = R"(
    namespace nx {
    /// Foo comment.
    int ff(int x, double y) { return 0; }
    }
  )";
  runSymbolCollector(Header, /*Main=*/"");
  EXPECT_THAT(Symbols,
              UnorderedElementsAre(QName("nx"),
                                   AllOf(QName("nx::ff"),
                                         Labeled("ff(int x, double y)"),
                                         Detail("int"), Doc("Foo comment."))));
}

TEST_F(SymbolCollectorTest, PlainAndSnippet) {
  const std::string Header = R"(
    namespace nx {
    void f() {}
    int ff(int x, double y) { return 0; }
    }
  )";
  runSymbolCollector(Header, /*Main=*/"");
  EXPECT_THAT(
      Symbols,
      UnorderedElementsAre(
          QName("nx"),
          AllOf(QName("nx::f"), Labeled("f()"), Plain("f"), Snippet("f()")),
          AllOf(QName("nx::ff"), Labeled("ff(int x, double y)"), Plain("ff"),
                Snippet("ff(${1:int x}, ${2:double y})"))));
}

TEST_F(SymbolCollectorTest, YAMLConversions) {
  const std::string YAML1 = R"(
---
ID: 057557CEBF6E6B2DD437FBF60CC58F352D1DF856
Name:   'Foo1'
Scope:   'clang::'
SymInfo:
  Kind:            Function
  Lang:            Cpp
CanonicalDeclaration:
  StartOffset:     0
  EndOffset:       1
  FilePath:        /path/foo.h
CompletionLabel:    'Foo1-label'
CompletionFilterText:    'filter'
CompletionPlainInsertText:    'plain'
Detail:
  Documentation:    'Foo doc'
  CompletionDetail:    'int'
...
)";
  const std::string YAML2 = R"(
---
ID: 057557CEBF6E6B2DD437FBF60CC58F352D1DF858
Name:   'Foo2'
Scope:   'clang::'
SymInfo:
  Kind:            Function
  Lang:            Cpp
CanonicalDeclaration:
  StartOffset:     10
  EndOffset:       12
  FilePath:        /path/foo.h
CompletionLabel:    'Foo2-label'
CompletionFilterText:    'filter'
CompletionPlainInsertText:    'plain'
CompletionSnippetInsertText:    'snippet'
...
)";

  auto Symbols1 = SymbolFromYAML(YAML1);
  EXPECT_THAT(Symbols1, UnorderedElementsAre(
                            AllOf(QName("clang::Foo1"), Labeled("Foo1-label"),
                                  Doc("Foo doc"), Detail("int"))));
  auto Symbols2 = SymbolFromYAML(YAML2);
  EXPECT_THAT(Symbols2, UnorderedElementsAre(AllOf(QName("clang::Foo2"),
                                                   Labeled("Foo2-label"),
                                                   Not(HasDetail()))));

  std::string ConcatenatedYAML =
      SymbolsToYAML(Symbols1) + SymbolsToYAML(Symbols2);
  auto ConcatenatedSymbols = SymbolFromYAML(ConcatenatedYAML);
  EXPECT_THAT(ConcatenatedSymbols,
              UnorderedElementsAre(QName("clang::Foo1"),
                                   QName("clang::Foo2")));
}

} // namespace
} // namespace clangd
} // namespace clang
