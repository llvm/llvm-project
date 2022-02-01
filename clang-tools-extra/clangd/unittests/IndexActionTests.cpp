//===------ IndexActionTests.cpp  -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Headers.h"
#include "TestFS.h"
#include "index/IndexAction.h"
#include "clang/Tooling/Tooling.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::EndsWith;
using ::testing::Not;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;
using ::testing::UnorderedPointwise;

std::string toUri(llvm::StringRef Path) { return URI::create(Path).toString(); }

MATCHER(isTU, "") { return arg.Flags & IncludeGraphNode::SourceFlag::IsTU; }

MATCHER_P(hasDigest, Digest, "") { return arg.Digest == Digest; }

MATCHER_P(hasName, Name, "") { return arg.Name == Name; }

MATCHER(hasSameURI, "") {
  llvm::StringRef URI = ::testing::get<0>(arg);
  const std::string &Path = ::testing::get<1>(arg);
  return toUri(Path) == URI;
}

::testing::Matcher<const IncludeGraphNode &>
includesAre(const std::vector<std::string> &Includes) {
  return ::testing::Field(&IncludeGraphNode::DirectIncludes,
                          UnorderedPointwise(hasSameURI(), Includes));
}

void checkNodesAreInitialized(const IndexFileIn &IndexFile,
                              const std::vector<std::string> &Paths) {
  ASSERT_TRUE(IndexFile.Sources);
  EXPECT_THAT(Paths.size(), IndexFile.Sources->size());
  for (llvm::StringRef Path : Paths) {
    auto URI = toUri(Path);
    const auto &Node = IndexFile.Sources->lookup(URI);
    // Uninitialized nodes will have an empty URI.
    EXPECT_EQ(Node.URI.data(), IndexFile.Sources->find(URI)->getKeyData());
  }
}

std::map<std::string, const IncludeGraphNode &> toMap(const IncludeGraph &IG) {
  std::map<std::string, const IncludeGraphNode &> Nodes;
  for (auto &I : IG)
    Nodes.emplace(std::string(I.getKey()), I.getValue());
  return Nodes;
}

class IndexActionTest : public ::testing::Test {
public:
  IndexActionTest() : InMemoryFileSystem(new llvm::vfs::InMemoryFileSystem) {}

  IndexFileIn
  runIndexingAction(llvm::StringRef MainFilePath,
                    const std::vector<std::string> &ExtraArgs = {}) {
    IndexFileIn IndexFile;
    llvm::IntrusiveRefCntPtr<FileManager> Files(
        new FileManager(FileSystemOptions(), InMemoryFileSystem));

    auto Action = createStaticIndexingAction(
        Opts, [&](SymbolSlab S) { IndexFile.Symbols = std::move(S); },
        [&](RefSlab R) { IndexFile.Refs = std::move(R); },
        [&](RelationSlab R) { IndexFile.Relations = std::move(R); },
        [&](IncludeGraph IG) { IndexFile.Sources = std::move(IG); });

    std::vector<std::string> Args = {"index_action", "-fsyntax-only",
                                     "-xc++",        "-std=c++11",
                                     "-iquote",      testRoot()};
    Args.insert(Args.end(), ExtraArgs.begin(), ExtraArgs.end());
    Args.push_back(std::string(MainFilePath));

    tooling::ToolInvocation Invocation(
        Args, std::move(Action), Files.get(),
        std::make_shared<PCHContainerOperations>());

    Invocation.run();

    checkNodesAreInitialized(IndexFile, FilePaths);
    return IndexFile;
  }

  void addFile(llvm::StringRef Path, llvm::StringRef Content) {
    InMemoryFileSystem->addFile(Path, 0,
                                llvm::MemoryBuffer::getMemBufferCopy(Content));
    FilePaths.push_back(std::string(Path));
  }

protected:
  SymbolCollector::Options Opts;
  std::vector<std::string> FilePaths;
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem;
};

TEST_F(IndexActionTest, CollectIncludeGraph) {
  std::string MainFilePath = testPath("main.cpp");
  std::string MainCode = "#include \"level1.h\"";
  std::string Level1HeaderPath = testPath("level1.h");
  std::string Level1HeaderCode = "#include \"level2.h\"";
  std::string Level2HeaderPath = testPath("level2.h");
  std::string Level2HeaderCode = "";

  addFile(MainFilePath, MainCode);
  addFile(Level1HeaderPath, Level1HeaderCode);
  addFile(Level2HeaderPath, Level2HeaderCode);

  IndexFileIn IndexFile = runIndexingAction(MainFilePath);
  auto Nodes = toMap(*IndexFile.Sources);

  EXPECT_THAT(Nodes,
              UnorderedElementsAre(
                  Pair(toUri(MainFilePath),
                       AllOf(isTU(), includesAre({Level1HeaderPath}),
                             hasDigest(digest(MainCode)))),
                  Pair(toUri(Level1HeaderPath),
                       AllOf(Not(isTU()), includesAre({Level2HeaderPath}),
                             hasDigest(digest(Level1HeaderCode)))),
                  Pair(toUri(Level2HeaderPath),
                       AllOf(Not(isTU()), includesAre({}),
                             hasDigest(digest(Level2HeaderCode))))));
}

TEST_F(IndexActionTest, IncludeGraphSelfInclude) {
  std::string MainFilePath = testPath("main.cpp");
  std::string MainCode = "#include \"header.h\"";
  std::string HeaderPath = testPath("header.h");
  std::string HeaderCode = R"cpp(
      #ifndef _GUARD_
      #define _GUARD_
      #include "header.h"
      #endif)cpp";

  addFile(MainFilePath, MainCode);
  addFile(HeaderPath, HeaderCode);

  IndexFileIn IndexFile = runIndexingAction(MainFilePath);
  auto Nodes = toMap(*IndexFile.Sources);

  EXPECT_THAT(
      Nodes,
      UnorderedElementsAre(
          Pair(toUri(MainFilePath), AllOf(isTU(), includesAre({HeaderPath}),
                                          hasDigest(digest(MainCode)))),
          Pair(toUri(HeaderPath), AllOf(Not(isTU()), includesAre({HeaderPath}),
                                        hasDigest(digest(HeaderCode))))));
}

TEST_F(IndexActionTest, IncludeGraphSkippedFile) {
  std::string MainFilePath = testPath("main.cpp");
  std::string MainCode = R"cpp(
      #include "common.h"
      #include "header.h"
      )cpp";

  std::string CommonHeaderPath = testPath("common.h");
  std::string CommonHeaderCode = R"cpp(
      #ifndef _GUARD_
      #define _GUARD_
      void f();
      #endif)cpp";

  std::string HeaderPath = testPath("header.h");
  std::string HeaderCode = R"cpp(
      #include "common.h"
      void g();)cpp";

  addFile(MainFilePath, MainCode);
  addFile(HeaderPath, HeaderCode);
  addFile(CommonHeaderPath, CommonHeaderCode);

  IndexFileIn IndexFile = runIndexingAction(MainFilePath);
  auto Nodes = toMap(*IndexFile.Sources);

  EXPECT_THAT(
      Nodes, UnorderedElementsAre(
                 Pair(toUri(MainFilePath),
                      AllOf(isTU(), includesAre({HeaderPath, CommonHeaderPath}),
                            hasDigest(digest(MainCode)))),
                 Pair(toUri(HeaderPath),
                      AllOf(Not(isTU()), includesAre({CommonHeaderPath}),
                            hasDigest(digest(HeaderCode)))),
                 Pair(toUri(CommonHeaderPath),
                      AllOf(Not(isTU()), includesAre({}),
                            hasDigest(digest(CommonHeaderCode))))));
}

TEST_F(IndexActionTest, IncludeGraphDynamicInclude) {
  std::string MainFilePath = testPath("main.cpp");
  std::string MainCode = R"cpp(
      #ifndef FOO
      #define FOO "main.cpp"
      #else
      #define FOO "header.h"
      #endif

      #include FOO)cpp";
  std::string HeaderPath = testPath("header.h");
  std::string HeaderCode = "";

  addFile(MainFilePath, MainCode);
  addFile(HeaderPath, HeaderCode);

  IndexFileIn IndexFile = runIndexingAction(MainFilePath);
  auto Nodes = toMap(*IndexFile.Sources);

  EXPECT_THAT(
      Nodes,
      UnorderedElementsAre(
          Pair(toUri(MainFilePath),
               AllOf(isTU(), includesAre({MainFilePath, HeaderPath}),
                     hasDigest(digest(MainCode)))),
          Pair(toUri(HeaderPath), AllOf(Not(isTU()), includesAre({}),
                                        hasDigest(digest(HeaderCode))))));
}

TEST_F(IndexActionTest, NoWarnings) {
  std::string MainFilePath = testPath("main.cpp");
  std::string MainCode = R"cpp(
      void foo(int x) {
        if (x = 1) // -Wparentheses
          return;
        if (x = 1) // -Wparentheses
          return;
      }
      void bar() {}
  )cpp";
  addFile(MainFilePath, MainCode);
  // We set -ferror-limit so the warning-promoted-to-error would be fatal.
  // This would cause indexing to stop (if warnings weren't disabled).
  IndexFileIn IndexFile = runIndexingAction(
      MainFilePath, {"-ferror-limit=1", "-Wparentheses", "-Werror"});
  ASSERT_TRUE(IndexFile.Sources);
  ASSERT_NE(0u, IndexFile.Sources->size());
  EXPECT_THAT(*IndexFile.Symbols, ElementsAre(hasName("foo"), hasName("bar")));
}

TEST_F(IndexActionTest, SkipFiles) {
  std::string MainFilePath = testPath("main.cpp");
  addFile(MainFilePath, R"cpp(
    // clang-format off
    #include "good.h"
    #include "bad.h"
    // clang-format on
  )cpp");
  addFile(testPath("good.h"), R"cpp(
    struct S { int s; };
    void f1() { S f; }
    auto unskippable1() { return S(); }
  )cpp");
  addFile(testPath("bad.h"), R"cpp(
    struct T { S t; };
    void f2() { S f; }
    auto unskippable2() { return S(); }
  )cpp");
  Opts.FileFilter = [](const SourceManager &SM, FileID F) {
    return !SM.getFileEntryForID(F)->getName().endswith("bad.h");
  };
  IndexFileIn IndexFile = runIndexingAction(MainFilePath, {"-std=c++14"});
  EXPECT_THAT(*IndexFile.Symbols,
              UnorderedElementsAre(hasName("S"), hasName("s"), hasName("f1"),
                                   hasName("unskippable1")));
  for (const auto &Pair : *IndexFile.Refs)
    for (const auto &Ref : Pair.second)
      EXPECT_THAT(Ref.Location.FileURI, EndsWith("good.h"));
}

TEST_F(IndexActionTest, SkipNestedSymbols) {
  std::string MainFilePath = testPath("main.cpp");
  addFile(MainFilePath, R"cpp(
  namespace ns1 {
  namespace ns2 {
  namespace ns3 {
  namespace ns4 {
  namespace ns5 {
  namespace ns6 {
  namespace ns7 {
  namespace ns8 {
  namespace ns9 {
  class Bar {};
  void foo() {
    class Baz {};
  }
  }
  }
  }
  }
  }
  }
  }
  }
  })cpp");
  IndexFileIn IndexFile = runIndexingAction(MainFilePath, {"-std=c++14"});
  EXPECT_THAT(*IndexFile.Symbols, testing::Contains(hasName("foo")));
  EXPECT_THAT(*IndexFile.Symbols, testing::Contains(hasName("Bar")));
  EXPECT_THAT(*IndexFile.Symbols, Not(testing::Contains(hasName("Baz"))));
}
} // namespace
} // namespace clangd
} // namespace clang
