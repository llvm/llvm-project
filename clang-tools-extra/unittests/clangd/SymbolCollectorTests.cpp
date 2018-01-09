//===-- SymbolCollectorTests.cpp  -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "index/SymbolCollector.h"
#include "index/SymbolYAML.h"
#include "clang/Index/IndexingAction.h"
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

using testing::Eq;
using testing::Field;
using testing::UnorderedElementsAre;
using testing::UnorderedElementsAreArray;

// GMock helpers for matching Symbol.
MATCHER_P(QName, Name, "") {
  return (arg.Scope + (arg.Scope.empty() ? "" : "::") + arg.Name).str() == Name;
}

namespace clang {
namespace clangd {

namespace {
class SymbolIndexActionFactory : public tooling::FrontendActionFactory {
public:
  SymbolIndexActionFactory() = default;

  clang::FrontendAction *create() override {
    index::IndexingOptions IndexOpts;
    IndexOpts.SystemSymbolFilter =
        index::IndexingOptions::SystemSymbolFilterKind::All;
    IndexOpts.IndexFunctionLocals = false;
    Collector = std::make_shared<SymbolCollector>();
    FrontendAction *Action =
        index::createIndexingAction(Collector, IndexOpts, nullptr).release();
    return Action;
  }

  std::shared_ptr<SymbolCollector> Collector;
};

class SymbolCollectorTest : public ::testing::Test {
public:
  bool runSymbolCollector(StringRef HeaderCode, StringRef MainCode) {
    llvm::IntrusiveRefCntPtr<vfs::InMemoryFileSystem> InMemoryFileSystem(
        new vfs::InMemoryFileSystem);
    llvm::IntrusiveRefCntPtr<FileManager> Files(
        new FileManager(FileSystemOptions(), InMemoryFileSystem));

    const std::string FileName = "symbol.cc";
    const std::string HeaderName = "symbols.h";
    auto Factory = llvm::make_unique<SymbolIndexActionFactory>();

    tooling::ToolInvocation Invocation(
        {"symbol_collector", "-fsyntax-only", "-std=c++11", FileName},
        Factory->create(), Files.get(),
        std::make_shared<PCHContainerOperations>());

    InMemoryFileSystem->addFile(HeaderName, 0,
                                llvm::MemoryBuffer::getMemBuffer(HeaderCode));

    std::string Content = "#include\"" + std::string(HeaderName) + "\"";
    Content += "\n" + MainCode.str();
    InMemoryFileSystem->addFile(FileName, 0,
                                llvm::MemoryBuffer::getMemBuffer(Content));
    Invocation.run();
    Symbols = Factory->Collector->takeSymbols();
    return true;
  }

protected:
  SymbolSlab Symbols;
};

TEST_F(SymbolCollectorTest, CollectSymbol) {
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
                  {QName("Foo"),
                   QName("Foo::f"),
                   QName("f1"),
                   QName("f2"),
                   QName("KInt"),
                   QName("kStr"),
                   QName("foo"),
                   QName("foo::bar"),
                   QName("foo::int32"),
                   QName("foo::int32_t"),
                   QName("foo::v1"),
                   QName("foo::bar::v2"),
                   QName("foo::baz")}));
}

TEST_F(SymbolCollectorTest, YAMLConversions) {
  const std::string YAML1 = R"(
---
ID: 057557CEBF6E6B2DD437FBF60CC58F352D1DF856
Name:   'Foo1'
Scope:   'clang'
SymInfo:
  Kind:            Function
  Lang:            Cpp
CanonicalDeclaration:
  StartOffset:     0
  EndOffset:       1
  FilePath:        /path/foo.h
...
)";
  const std::string YAML2 = R"(
---
ID: 057557CEBF6E6B2DD437FBF60CC58F352D1DF858
Name:   'Foo2'
Scope:   'clang'
SymInfo:
  Kind:            Function
  Lang:            Cpp
CanonicalDeclaration:
  StartOffset:     10
  EndOffset:       12
  FilePath:        /path/foo.h
...
)";

  auto Symbols1 = SymbolFromYAML(YAML1);
  EXPECT_THAT(Symbols1,
              UnorderedElementsAre(QName("clang::Foo1")));
  auto Symbols2 = SymbolFromYAML(YAML2);
  EXPECT_THAT(Symbols2,
              UnorderedElementsAre(QName("clang::Foo2")));

  std::string ConcatenatedYAML =
      SymbolToYAML(Symbols1) + SymbolToYAML(Symbols2);
  auto ConcatenatedSymbols = SymbolFromYAML(ConcatenatedYAML);
  EXPECT_THAT(ConcatenatedSymbols,
              UnorderedElementsAre(QName("clang::Foo1"),
                                   QName("clang::Foo2")));
}

} // namespace
} // namespace clangd
} // namespace clang
