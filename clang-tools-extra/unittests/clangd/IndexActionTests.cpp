//===------ IndexActionTests.cpp  -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

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
using ::testing::Not;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;
using ::testing::UnorderedPointwise;

std::string toUri(llvm::StringRef Path) { return URI::create(Path).toString(); }

MATCHER(IsTU, "") { return arg.IsTU; }

MATCHER_P(HasDigest, Digest, "") { return arg.Digest == Digest; }

MATCHER(HasSameURI, "") {
  llvm::StringRef URI = testing::get<0>(arg);
  const std::string &Path = testing::get<1>(arg);
  return toUri(Path) == URI;
}

testing::Matcher<const IncludeGraphNode &>
IncludesAre(const std::vector<std::string> &Includes) {
  return ::testing::Field(&IncludeGraphNode::DirectIncludes,
                          UnorderedPointwise(HasSameURI(), Includes));
}

void checkNodesAreInitialized(const IndexFileIn &IndexFile,
                              const std::vector<std::string> &Paths) {
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
    Nodes.emplace(I.getKey(), I.getValue());
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
        SymbolCollector::Options(),
        [&](SymbolSlab S) { IndexFile.Symbols = std::move(S); },
        [&](RefSlab R) { IndexFile.Refs = std::move(R); },
        [&](IncludeGraph IG) { IndexFile.Sources = std::move(IG); });

    std::vector<std::string> Args = {"index_action", "-fsyntax-only",
                                     "-xc++",        "-std=c++11",
                                     "-iquote",      testRoot()};
    Args.insert(Args.end(), ExtraArgs.begin(), ExtraArgs.end());
    Args.push_back(MainFilePath);

    tooling::ToolInvocation Invocation(
        Args, Action.release(), Files.get(),
        std::make_shared<PCHContainerOperations>());

    Invocation.run();

    checkNodesAreInitialized(IndexFile, FilePaths);
    return IndexFile;
  }

  void addFile(llvm::StringRef Path, llvm::StringRef Content) {
    InMemoryFileSystem->addFile(Path, 0,
                                llvm::MemoryBuffer::getMemBuffer(Content));
    FilePaths.push_back(Path);
  }

protected:
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
                       AllOf(IsTU(), IncludesAre({Level1HeaderPath}),
                             HasDigest(digest(MainCode)))),
                  Pair(toUri(Level1HeaderPath),
                       AllOf(Not(IsTU()), IncludesAre({Level2HeaderPath}),
                             HasDigest(digest(Level1HeaderCode)))),
                  Pair(toUri(Level2HeaderPath),
                       AllOf(Not(IsTU()), IncludesAre({}),
                             HasDigest(digest(Level2HeaderCode))))));
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
          Pair(toUri(MainFilePath), AllOf(IsTU(), IncludesAre({HeaderPath}),
                                          HasDigest(digest(MainCode)))),
          Pair(toUri(HeaderPath), AllOf(Not(IsTU()), IncludesAre({HeaderPath}),
                                        HasDigest(digest(HeaderCode))))));
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
                      AllOf(IsTU(), IncludesAre({HeaderPath, CommonHeaderPath}),
                            HasDigest(digest(MainCode)))),
                 Pair(toUri(HeaderPath),
                      AllOf(Not(IsTU()), IncludesAre({CommonHeaderPath}),
                            HasDigest(digest(HeaderCode)))),
                 Pair(toUri(CommonHeaderPath),
                      AllOf(Not(IsTU()), IncludesAre({}),
                            HasDigest(digest(CommonHeaderCode))))));
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
               AllOf(IsTU(), IncludesAre({MainFilePath, HeaderPath}),
                     HasDigest(digest(MainCode)))),
          Pair(toUri(HeaderPath), AllOf(Not(IsTU()), IncludesAre({}),
                                        HasDigest(digest(HeaderCode))))));
}

} // namespace
} // namespace clangd
} // namespace clang
