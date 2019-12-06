#include "Headers.h"
#include "SyncAPI.h"
#include "TestFS.h"
#include "TestIndex.h"
#include "TestTU.h"
#include "index/Background.h"
#include "index/BackgroundRebuild.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/Threading.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <deque>
#include <thread>

using ::testing::_;
using ::testing::AllOf;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::Not;
using ::testing::UnorderedElementsAre;

namespace clang {
namespace clangd {

MATCHER_P(Named, N, "") { return arg.Name == N; }
MATCHER(Declared, "") {
  return !StringRef(arg.CanonicalDeclaration.FileURI).empty();
}
MATCHER(Defined, "") { return !StringRef(arg.Definition.FileURI).empty(); }
MATCHER_P(FileURI, F, "") { return StringRef(arg.Location.FileURI) == F; }
::testing::Matcher<const RefSlab &>
RefsAre(std::vector<::testing::Matcher<Ref>> Matchers) {
  return ElementsAre(::testing::Pair(_, UnorderedElementsAreArray(Matchers)));
}
// URI cannot be empty since it references keys in the IncludeGraph.
MATCHER(EmptyIncludeNode, "") {
  return arg.Flags == IncludeGraphNode::SourceFlag::None && !arg.URI.empty() &&
         arg.Digest == FileDigest{{0}} && arg.DirectIncludes.empty();
}

MATCHER(HadErrors, "") {
  return arg.Flags & IncludeGraphNode::SourceFlag::HadErrors;
}

MATCHER_P(NumReferences, N, "") { return arg.References == N; }

class MemoryShardStorage : public BackgroundIndexStorage {
  mutable std::mutex StorageMu;
  llvm::StringMap<std::string> &Storage;
  size_t &CacheHits;

public:
  MemoryShardStorage(llvm::StringMap<std::string> &Storage, size_t &CacheHits)
      : Storage(Storage), CacheHits(CacheHits) {}
  llvm::Error storeShard(llvm::StringRef ShardIdentifier,
                         IndexFileOut Shard) const override {
    std::lock_guard<std::mutex> Lock(StorageMu);
    AccessedPaths.insert(ShardIdentifier);
    Storage[ShardIdentifier] = llvm::to_string(Shard);
    return llvm::Error::success();
  }
  std::unique_ptr<IndexFileIn>
  loadShard(llvm::StringRef ShardIdentifier) const override {
    std::lock_guard<std::mutex> Lock(StorageMu);
    AccessedPaths.insert(ShardIdentifier);
    if (Storage.find(ShardIdentifier) == Storage.end()) {
      return nullptr;
    }
    auto IndexFile = readIndexFile(Storage[ShardIdentifier]);
    if (!IndexFile) {
      ADD_FAILURE() << "Error while reading " << ShardIdentifier << ':'
                    << IndexFile.takeError();
      return nullptr;
    }
    CacheHits++;
    return std::make_unique<IndexFileIn>(std::move(*IndexFile));
  }

  mutable llvm::StringSet<> AccessedPaths;
};

class BackgroundIndexTest : public ::testing::Test {
protected:
  BackgroundIndexTest() { BackgroundQueue::preventThreadStarvationInTests(); }
};

TEST_F(BackgroundIndexTest, NoCrashOnErrorFile) {
  MockFSProvider FS;
  FS.Files[testPath("root/A.cc")] = "error file";
  llvm::StringMap<std::string> Storage;
  size_t CacheHits = 0;
  MemoryShardStorage MSS(Storage, CacheHits);
  OverlayCDB CDB(/*Base=*/nullptr);
  BackgroundIndex Idx(Context::empty(), FS, CDB,
                      [&](llvm::StringRef) { return &MSS; });

  tooling::CompileCommand Cmd;
  Cmd.Filename = testPath("root/A.cc");
  Cmd.Directory = testPath("root");
  Cmd.CommandLine = {"clang++", "-DA=1", testPath("root/A.cc")};
  CDB.setCompileCommand(testPath("root/A.cc"), Cmd);

  ASSERT_TRUE(Idx.blockUntilIdleForTest());
}

TEST_F(BackgroundIndexTest, IndexTwoFiles) {
  MockFSProvider FS;
  // a.h yields different symbols when included by A.cc vs B.cc.
  FS.Files[testPath("root/A.h")] = R"cpp(
      void common();
      void f_b();
      #if A
        class A_CC {};
      #else
        class B_CC{};
      #endif
      )cpp";
  FS.Files[testPath("root/A.cc")] =
      "#include \"A.h\"\nvoid g() { (void)common; }";
  FS.Files[testPath("root/B.cc")] =
      R"cpp(
      #define A 0
      #include "A.h"
      void f_b() {
        (void)common;
        (void)common;
        (void)common;
        (void)common;
      })cpp";
  llvm::StringMap<std::string> Storage;
  size_t CacheHits = 0;
  MemoryShardStorage MSS(Storage, CacheHits);
  OverlayCDB CDB(/*Base=*/nullptr);
  BackgroundIndex Idx(Context::empty(), FS, CDB,
                      [&](llvm::StringRef) { return &MSS; });

  tooling::CompileCommand Cmd;
  Cmd.Filename = testPath("root/A.cc");
  Cmd.Directory = testPath("root");
  Cmd.CommandLine = {"clang++", "-DA=1", testPath("root/A.cc")};
  CDB.setCompileCommand(testPath("root/A.cc"), Cmd);

  ASSERT_TRUE(Idx.blockUntilIdleForTest());
  EXPECT_THAT(runFuzzyFind(Idx, ""),
              UnorderedElementsAre(AllOf(Named("common"), NumReferences(1U)),
                                   AllOf(Named("A_CC"), NumReferences(0U)),
                                   AllOf(Named("g"), NumReferences(0U)),
                                   AllOf(Named("f_b"), Declared(),
                                         Not(Defined()), NumReferences(0U))));

  Cmd.Filename = testPath("root/B.cc");
  Cmd.CommandLine = {"clang++", Cmd.Filename};
  CDB.setCompileCommand(testPath("root/B.cc"), Cmd);

  ASSERT_TRUE(Idx.blockUntilIdleForTest());
  // B_CC is dropped as we don't collect symbols from A.h in this compilation.
  EXPECT_THAT(runFuzzyFind(Idx, ""),
              UnorderedElementsAre(AllOf(Named("common"), NumReferences(5U)),
                                   AllOf(Named("A_CC"), NumReferences(0U)),
                                   AllOf(Named("g"), NumReferences(0U)),
                                   AllOf(Named("f_b"), Declared(), Defined(),
                                         NumReferences(1U))));

  auto Syms = runFuzzyFind(Idx, "common");
  EXPECT_THAT(Syms, UnorderedElementsAre(Named("common")));
  auto Common = *Syms.begin();
  EXPECT_THAT(getRefs(Idx, Common.ID),
              RefsAre({FileURI("unittest:///root/A.h"),
                       FileURI("unittest:///root/A.cc"),
                       FileURI("unittest:///root/B.cc"),
                       FileURI("unittest:///root/B.cc"),
                       FileURI("unittest:///root/B.cc"),
                       FileURI("unittest:///root/B.cc")}));
}

TEST_F(BackgroundIndexTest, ShardStorageTest) {
  MockFSProvider FS;
  FS.Files[testPath("root/A.h")] = R"cpp(
      void common();
      void f_b();
      class A_CC {};
      )cpp";
  std::string A_CC = "";
  FS.Files[testPath("root/A.cc")] = R"cpp(
      #include "A.h"
      void g() { (void)common; }
      class B_CC : public A_CC {};
      )cpp";

  llvm::StringMap<std::string> Storage;
  size_t CacheHits = 0;
  MemoryShardStorage MSS(Storage, CacheHits);

  tooling::CompileCommand Cmd;
  Cmd.Filename = testPath("root/A.cc");
  Cmd.Directory = testPath("root");
  Cmd.CommandLine = {"clang++", testPath("root/A.cc")};
  // Check nothing is loaded from Storage, but A.cc and A.h has been stored.
  {
    OverlayCDB CDB(/*Base=*/nullptr);
    BackgroundIndex Idx(Context::empty(), FS, CDB,
                        [&](llvm::StringRef) { return &MSS; });
    CDB.setCompileCommand(testPath("root/A.cc"), Cmd);
    ASSERT_TRUE(Idx.blockUntilIdleForTest());
  }
  EXPECT_EQ(CacheHits, 0U);
  EXPECT_EQ(Storage.size(), 2U);

  {
    OverlayCDB CDB(/*Base=*/nullptr);
    BackgroundIndex Idx(Context::empty(), FS, CDB,
                        [&](llvm::StringRef) { return &MSS; });
    CDB.setCompileCommand(testPath("root/A.cc"), Cmd);
    ASSERT_TRUE(Idx.blockUntilIdleForTest());
  }
  EXPECT_EQ(CacheHits, 2U); // Check both A.cc and A.h loaded from cache.
  EXPECT_EQ(Storage.size(), 2U);

  auto ShardHeader = MSS.loadShard(testPath("root/A.h"));
  EXPECT_NE(ShardHeader, nullptr);
  EXPECT_THAT(
      *ShardHeader->Symbols,
      UnorderedElementsAre(Named("common"), Named("A_CC"),
                           AllOf(Named("f_b"), Declared(), Not(Defined()))));
  for (const auto &Ref : *ShardHeader->Refs)
    EXPECT_THAT(Ref.second,
                UnorderedElementsAre(FileURI("unittest:///root/A.h")));

  auto ShardSource = MSS.loadShard(testPath("root/A.cc"));
  EXPECT_NE(ShardSource, nullptr);
  EXPECT_THAT(*ShardSource->Symbols,
              UnorderedElementsAre(Named("g"), Named("B_CC")));
  for (const auto &Ref : *ShardSource->Refs)
    EXPECT_THAT(Ref.second,
                UnorderedElementsAre(FileURI("unittest:///root/A.cc")));

  // The BaseOf relationship between A_CC and B_CC is stored in the file
  // containing the definition of the subject (A_CC)
  SymbolID A = findSymbol(*ShardHeader->Symbols, "A_CC").ID;
  SymbolID B = findSymbol(*ShardSource->Symbols, "B_CC").ID;
  EXPECT_THAT(*ShardHeader->Relations,
              UnorderedElementsAre(Relation{A, RelationKind::BaseOf, B}));
  // (and not in the file containing the definition of the object (B_CC)).
  EXPECT_EQ(ShardSource->Relations->size(), 0u);
}

TEST_F(BackgroundIndexTest, DirectIncludesTest) {
  MockFSProvider FS;
  FS.Files[testPath("root/B.h")] = "";
  FS.Files[testPath("root/A.h")] = R"cpp(
      #include "B.h"
      void common();
      void f_b();
      class A_CC {};
      )cpp";
  std::string A_CC = "#include \"A.h\"\nvoid g() { (void)common; }";
  FS.Files[testPath("root/A.cc")] = A_CC;

  llvm::StringMap<std::string> Storage;
  size_t CacheHits = 0;
  MemoryShardStorage MSS(Storage, CacheHits);

  tooling::CompileCommand Cmd;
  Cmd.Filename = testPath("root/A.cc");
  Cmd.Directory = testPath("root");
  Cmd.CommandLine = {"clang++", testPath("root/A.cc")};
  {
    OverlayCDB CDB(/*Base=*/nullptr);
    BackgroundIndex Idx(Context::empty(), FS, CDB,
                        [&](llvm::StringRef) { return &MSS; });
    CDB.setCompileCommand(testPath("root/A.cc"), Cmd);
    ASSERT_TRUE(Idx.blockUntilIdleForTest());
  }

  auto ShardSource = MSS.loadShard(testPath("root/A.cc"));
  EXPECT_TRUE(ShardSource->Sources);
  EXPECT_EQ(ShardSource->Sources->size(), 2U); // A.cc, A.h
  EXPECT_THAT(
      ShardSource->Sources->lookup("unittest:///root/A.cc").DirectIncludes,
      UnorderedElementsAre("unittest:///root/A.h"));
  EXPECT_NE(ShardSource->Sources->lookup("unittest:///root/A.cc").Digest,
            FileDigest{{0}});
  EXPECT_THAT(ShardSource->Sources->lookup("unittest:///root/A.h"),
              EmptyIncludeNode());

  auto ShardHeader = MSS.loadShard(testPath("root/A.h"));
  EXPECT_TRUE(ShardHeader->Sources);
  EXPECT_EQ(ShardHeader->Sources->size(), 2U); // A.h, B.h
  EXPECT_THAT(
      ShardHeader->Sources->lookup("unittest:///root/A.h").DirectIncludes,
      UnorderedElementsAre("unittest:///root/B.h"));
  EXPECT_NE(ShardHeader->Sources->lookup("unittest:///root/A.h").Digest,
            FileDigest{{0}});
  EXPECT_THAT(ShardHeader->Sources->lookup("unittest:///root/B.h"),
              EmptyIncludeNode());
}

TEST_F(BackgroundIndexTest, ShardStorageLoad) {
  MockFSProvider FS;
  FS.Files[testPath("root/A.h")] = R"cpp(
      void common();
      void f_b();
      class A_CC {};
      )cpp";
  FS.Files[testPath("root/A.cc")] =
      "#include \"A.h\"\nvoid g() { (void)common; }";

  llvm::StringMap<std::string> Storage;
  size_t CacheHits = 0;
  MemoryShardStorage MSS(Storage, CacheHits);

  tooling::CompileCommand Cmd;
  Cmd.Filename = testPath("root/A.cc");
  Cmd.Directory = testPath("root");
  Cmd.CommandLine = {"clang++", testPath("root/A.cc")};
  // Check nothing is loaded from Storage, but A.cc and A.h has been stored.
  {
    OverlayCDB CDB(/*Base=*/nullptr);
    BackgroundIndex Idx(Context::empty(), FS, CDB,
                        [&](llvm::StringRef) { return &MSS; });
    CDB.setCompileCommand(testPath("root/A.cc"), Cmd);
    ASSERT_TRUE(Idx.blockUntilIdleForTest());
  }

  // Change header.
  FS.Files[testPath("root/A.h")] = R"cpp(
      void common();
      void f_b();
      class A_CC {};
      class A_CCnew {};
      )cpp";
  {
    OverlayCDB CDB(/*Base=*/nullptr);
    BackgroundIndex Idx(Context::empty(), FS, CDB,
                        [&](llvm::StringRef) { return &MSS; });
    CDB.setCompileCommand(testPath("root/A.cc"), Cmd);
    ASSERT_TRUE(Idx.blockUntilIdleForTest());
  }
  EXPECT_EQ(CacheHits, 2U); // Check both A.cc and A.h loaded from cache.

  // Check if the new symbol has arrived.
  auto ShardHeader = MSS.loadShard(testPath("root/A.h"));
  EXPECT_NE(ShardHeader, nullptr);
  EXPECT_THAT(*ShardHeader->Symbols, Contains(Named("A_CCnew")));

  // Change source.
  FS.Files[testPath("root/A.cc")] =
      "#include \"A.h\"\nvoid g() { (void)common; }\nvoid f_b() {}";
  {
    CacheHits = 0;
    OverlayCDB CDB(/*Base=*/nullptr);
    BackgroundIndex Idx(Context::empty(), FS, CDB,
                        [&](llvm::StringRef) { return &MSS; });
    CDB.setCompileCommand(testPath("root/A.cc"), Cmd);
    ASSERT_TRUE(Idx.blockUntilIdleForTest());
  }
  EXPECT_EQ(CacheHits, 2U); // Check both A.cc and A.h loaded from cache.

  // Check if the new symbol has arrived.
  ShardHeader = MSS.loadShard(testPath("root/A.h"));
  EXPECT_NE(ShardHeader, nullptr);
  EXPECT_THAT(*ShardHeader->Symbols, Contains(Named("A_CCnew")));
  auto ShardSource = MSS.loadShard(testPath("root/A.cc"));
  EXPECT_NE(ShardSource, nullptr);
  EXPECT_THAT(*ShardSource->Symbols,
              Contains(AllOf(Named("f_b"), Declared(), Defined())));
}

TEST_F(BackgroundIndexTest, ShardStorageEmptyFile) {
  MockFSProvider FS;
  FS.Files[testPath("root/A.h")] = R"cpp(
      void common();
      void f_b();
      class A_CC {};
      )cpp";
  FS.Files[testPath("root/B.h")] = R"cpp(
      #include "A.h"
      )cpp";
  FS.Files[testPath("root/A.cc")] =
      "#include \"B.h\"\nvoid g() { (void)common; }";

  llvm::StringMap<std::string> Storage;
  size_t CacheHits = 0;
  MemoryShardStorage MSS(Storage, CacheHits);

  tooling::CompileCommand Cmd;
  Cmd.Filename = testPath("root/A.cc");
  Cmd.Directory = testPath("root");
  Cmd.CommandLine = {"clang++", testPath("root/A.cc")};
  // Check that A.cc, A.h and B.h has been stored.
  {
    OverlayCDB CDB(/*Base=*/nullptr);
    BackgroundIndex Idx(Context::empty(), FS, CDB,
                        [&](llvm::StringRef) { return &MSS; });
    CDB.setCompileCommand(testPath("root/A.cc"), Cmd);
    ASSERT_TRUE(Idx.blockUntilIdleForTest());
  }
  EXPECT_THAT(Storage.keys(),
              UnorderedElementsAre(testPath("root/A.cc"), testPath("root/A.h"),
                                   testPath("root/B.h")));
  auto ShardHeader = MSS.loadShard(testPath("root/B.h"));
  EXPECT_NE(ShardHeader, nullptr);
  EXPECT_TRUE(ShardHeader->Symbols->empty());

  // Check that A.cc, A.h and B.h has been loaded.
  {
    CacheHits = 0;
    OverlayCDB CDB(/*Base=*/nullptr);
    BackgroundIndex Idx(Context::empty(), FS, CDB,
                        [&](llvm::StringRef) { return &MSS; });
    CDB.setCompileCommand(testPath("root/A.cc"), Cmd);
    ASSERT_TRUE(Idx.blockUntilIdleForTest());
  }
  EXPECT_EQ(CacheHits, 3U);

  // Update B.h to contain some symbols.
  FS.Files[testPath("root/B.h")] = R"cpp(
      #include "A.h"
      void new_func();
      )cpp";
  // Check that B.h has been stored with new contents.
  {
    CacheHits = 0;
    OverlayCDB CDB(/*Base=*/nullptr);
    BackgroundIndex Idx(Context::empty(), FS, CDB,
                        [&](llvm::StringRef) { return &MSS; });
    CDB.setCompileCommand(testPath("root/A.cc"), Cmd);
    ASSERT_TRUE(Idx.blockUntilIdleForTest());
  }
  EXPECT_EQ(CacheHits, 3U);
  ShardHeader = MSS.loadShard(testPath("root/B.h"));
  EXPECT_NE(ShardHeader, nullptr);
  EXPECT_THAT(*ShardHeader->Symbols,
              Contains(AllOf(Named("new_func"), Declared(), Not(Defined()))));
}

TEST_F(BackgroundIndexTest, NoDotsInAbsPath) {
  MockFSProvider FS;
  llvm::StringMap<std::string> Storage;
  size_t CacheHits = 0;
  MemoryShardStorage MSS(Storage, CacheHits);
  OverlayCDB CDB(/*Base=*/nullptr);
  BackgroundIndex Idx(Context::empty(), FS, CDB,
                      [&](llvm::StringRef) { return &MSS; });

  tooling::CompileCommand Cmd;
  FS.Files[testPath("root/A.cc")] = "";
  Cmd.Filename = "../A.cc";
  Cmd.Directory = testPath("root/build");
  Cmd.CommandLine = {"clang++", "../A.cc"};
  CDB.setCompileCommand(testPath("root/build/../A.cc"), Cmd);

  FS.Files[testPath("root/B.cc")] = "";
  Cmd.Filename = "./B.cc";
  Cmd.Directory = testPath("root");
  Cmd.CommandLine = {"clang++", "./B.cc"};
  CDB.setCompileCommand(testPath("root/./B.cc"), Cmd);

  ASSERT_TRUE(Idx.blockUntilIdleForTest());
  for (llvm::StringRef AbsPath : MSS.AccessedPaths.keys()) {
    EXPECT_FALSE(AbsPath.contains("./")) << AbsPath;
    EXPECT_FALSE(AbsPath.contains("../")) << AbsPath;
  }
}

TEST_F(BackgroundIndexTest, UncompilableFiles) {
  MockFSProvider FS;
  llvm::StringMap<std::string> Storage;
  size_t CacheHits = 0;
  MemoryShardStorage MSS(Storage, CacheHits);
  OverlayCDB CDB(/*Base=*/nullptr);
  BackgroundIndex Idx(Context::empty(), FS, CDB,
                      [&](llvm::StringRef) { return &MSS; });

  tooling::CompileCommand Cmd;
  FS.Files[testPath("A.h")] = "void foo();";
  FS.Files[testPath("B.h")] = "#include \"C.h\"\nasdf;";
  FS.Files[testPath("C.h")] = "";
  FS.Files[testPath("A.cc")] = R"cpp(
  #include "A.h"
  #include "B.h"
  #include "not_found_header.h"

  void foo() {}
  )cpp";
  Cmd.Filename = "../A.cc";
  Cmd.Directory = testPath("build");
  Cmd.CommandLine = {"clang++", "../A.cc"};
  CDB.setCompileCommand(testPath("build/../A.cc"), Cmd);
  ASSERT_TRUE(Idx.blockUntilIdleForTest());

  EXPECT_THAT(Storage.keys(), ElementsAre(testPath("A.cc"), testPath("A.h"),
                                          testPath("B.h"), testPath("C.h")));

  {
    auto Shard = MSS.loadShard(testPath("A.cc"));
    EXPECT_THAT(*Shard->Symbols, UnorderedElementsAre(Named("foo")));
    EXPECT_THAT(Shard->Sources->keys(),
                UnorderedElementsAre("unittest:///A.cc", "unittest:///A.h",
                                     "unittest:///B.h"));
    EXPECT_THAT(Shard->Sources->lookup("unittest:///A.cc"), HadErrors());
  }

  {
    auto Shard = MSS.loadShard(testPath("A.h"));
    EXPECT_THAT(*Shard->Symbols, UnorderedElementsAre(Named("foo")));
    EXPECT_THAT(Shard->Sources->keys(),
                UnorderedElementsAre("unittest:///A.h"));
    EXPECT_THAT(Shard->Sources->lookup("unittest:///A.h"), HadErrors());
  }

  {
    auto Shard = MSS.loadShard(testPath("B.h"));
    EXPECT_THAT(*Shard->Symbols, UnorderedElementsAre(Named("asdf")));
    EXPECT_THAT(Shard->Sources->keys(),
                UnorderedElementsAre("unittest:///B.h", "unittest:///C.h"));
    EXPECT_THAT(Shard->Sources->lookup("unittest:///B.h"), HadErrors());
  }

  {
    auto Shard = MSS.loadShard(testPath("C.h"));
    EXPECT_THAT(*Shard->Symbols, UnorderedElementsAre());
    EXPECT_THAT(Shard->Sources->keys(),
                UnorderedElementsAre("unittest:///C.h"));
    EXPECT_THAT(Shard->Sources->lookup("unittest:///C.h"), HadErrors());
  }
}

TEST_F(BackgroundIndexTest, CmdLineHash) {
  MockFSProvider FS;
  llvm::StringMap<std::string> Storage;
  size_t CacheHits = 0;
  MemoryShardStorage MSS(Storage, CacheHits);
  OverlayCDB CDB(/*Base=*/nullptr);
  BackgroundIndex Idx(Context::empty(), FS, CDB,
                      [&](llvm::StringRef) { return &MSS; });

  tooling::CompileCommand Cmd;
  FS.Files[testPath("A.cc")] = "#include \"A.h\"";
  FS.Files[testPath("A.h")] = "";
  Cmd.Filename = "../A.cc";
  Cmd.Directory = testPath("build");
  Cmd.CommandLine = {"clang++", "../A.cc", "-fsyntax-only"};
  CDB.setCompileCommand(testPath("build/../A.cc"), Cmd);
  ASSERT_TRUE(Idx.blockUntilIdleForTest());

  EXPECT_THAT(Storage.keys(), ElementsAre(testPath("A.cc"), testPath("A.h")));
  // Make sure we only store the Cmd for main file.
  EXPECT_FALSE(MSS.loadShard(testPath("A.h"))->Cmd);

  {
    tooling::CompileCommand CmdStored = *MSS.loadShard(testPath("A.cc"))->Cmd;
    EXPECT_EQ(CmdStored.CommandLine, Cmd.CommandLine);
    EXPECT_EQ(CmdStored.Directory, Cmd.Directory);
  }

  // FIXME: Changing compile commands should be enough to invalidate the cache.
  FS.Files[testPath("A.cc")] = " ";
  Cmd.CommandLine = {"clang++", "../A.cc", "-Dfoo", "-fsyntax-only"};
  CDB.setCompileCommand(testPath("build/../A.cc"), Cmd);
  ASSERT_TRUE(Idx.blockUntilIdleForTest());

  EXPECT_FALSE(MSS.loadShard(testPath("A.h"))->Cmd);

  {
    tooling::CompileCommand CmdStored = *MSS.loadShard(testPath("A.cc"))->Cmd;
    EXPECT_EQ(CmdStored.CommandLine, Cmd.CommandLine);
    EXPECT_EQ(CmdStored.Directory, Cmd.Directory);
  }
}

class BackgroundIndexRebuilderTest : public testing::Test {
protected:
  BackgroundIndexRebuilderTest()
      : Target(std::make_unique<MemIndex>()),
        Rebuilder(&Target, &Source, /*Threads=*/10) {
    // Prepare FileSymbols with TestSymbol in it, for checkRebuild.
    TestSymbol.ID = SymbolID("foo");
  }

  // Perform Action and determine whether it rebuilt the index or not.
  bool checkRebuild(std::function<void()> Action) {
    // Update name so we can tell if the index updates.
    VersionStorage.push_back("Sym" + std::to_string(++VersionCounter));
    TestSymbol.Name = VersionStorage.back();
    SymbolSlab::Builder SB;
    SB.insert(TestSymbol);
    Source.update("", std::make_unique<SymbolSlab>(std::move(SB).build()),
                  nullptr, nullptr, false);
    // Now maybe update the index.
    Action();
    // Now query the index to get the name count.
    std::string ReadName;
    LookupRequest Req;
    Req.IDs.insert(TestSymbol.ID);
    Target.lookup(Req, [&](const Symbol &S) { ReadName = S.Name; });
    // The index was rebuild if the name is up to date.
    return ReadName == VersionStorage.back();
  }

  Symbol TestSymbol;
  FileSymbols Source;
  SwapIndex Target;
  BackgroundIndexRebuilder Rebuilder;

  unsigned VersionCounter = 0;
  std::deque<std::string> VersionStorage;
};

TEST_F(BackgroundIndexRebuilderTest, IndexingTUs) {
  for (unsigned I = 0; I < Rebuilder.TUsBeforeFirstBuild - 1; ++I)
    EXPECT_FALSE(checkRebuild([&] { Rebuilder.indexedTU(); }));
  EXPECT_TRUE(checkRebuild([&] { Rebuilder.indexedTU(); }));
  for (unsigned I = 0; I < Rebuilder.TUsBeforeRebuild - 1; ++I)
    EXPECT_FALSE(checkRebuild([&] { Rebuilder.indexedTU(); }));
  EXPECT_TRUE(checkRebuild([&] { Rebuilder.indexedTU(); }));
}

TEST_F(BackgroundIndexRebuilderTest, LoadingShards) {
  Rebuilder.startLoading();
  Rebuilder.loadedShard(10);
  Rebuilder.loadedShard(20);
  EXPECT_TRUE(checkRebuild([&] { Rebuilder.doneLoading(); }));

  // No rebuild for no shards.
  Rebuilder.startLoading();
  EXPECT_FALSE(checkRebuild([&] { Rebuilder.doneLoading(); }));

  // Loads can overlap.
  Rebuilder.startLoading();
  Rebuilder.loadedShard(1);
  Rebuilder.startLoading();
  Rebuilder.loadedShard(1);
  EXPECT_FALSE(checkRebuild([&] { Rebuilder.doneLoading(); }));
  Rebuilder.loadedShard(1);
  EXPECT_TRUE(checkRebuild([&] { Rebuilder.doneLoading(); }));

  // No rebuilding for indexed files while loading.
  Rebuilder.startLoading();
  for (unsigned I = 0; I < 3 * Rebuilder.TUsBeforeRebuild; ++I)
    EXPECT_FALSE(checkRebuild([&] { Rebuilder.indexedTU(); }));
  // But they get indexed when we're done, even if no shards were loaded.
  EXPECT_TRUE(checkRebuild([&] { Rebuilder.doneLoading(); }));
}

TEST(BackgroundQueueTest, Priority) {
  // Create high and low priority tasks.
  // Once a bunch of high priority tasks have run, the queue is stopped.
  // So the low priority tasks should never run.
  BackgroundQueue Q;
  std::atomic<unsigned> HiRan(0), LoRan(0);
  BackgroundQueue::Task Lo([&] { ++LoRan; });
  BackgroundQueue::Task Hi([&] {
    if (++HiRan >= 10)
      Q.stop();
  });
  Hi.QueuePri = 100;

  // Enqueuing the low-priority ones first shouldn't make them run first.
  Q.append(std::vector<BackgroundQueue::Task>(30, Lo));
  for (unsigned I = 0; I < 30; ++I)
    Q.push(Hi);

  AsyncTaskRunner ThreadPool;
  for (unsigned I = 0; I < 5; ++I)
    ThreadPool.runAsync("worker", [&] { Q.work(); });
  // We should test enqueue with active workers, but it's hard to avoid races.
  // Just make sure we don't crash.
  Q.push(Lo);
  Q.append(std::vector<BackgroundQueue::Task>(2, Hi));

  // After finishing, check the tasks that ran.
  ThreadPool.wait();
  EXPECT_GE(HiRan, 10u);
  EXPECT_EQ(LoRan, 0u);
}

TEST(BackgroundQueueTest, Boost) {
  std::string Sequence;

  BackgroundQueue::Task A([&] { Sequence.push_back('A'); });
  A.Tag = "A";
  A.QueuePri = 1;

  BackgroundQueue::Task B([&] { Sequence.push_back('B'); });
  B.QueuePri = 2;
  B.Tag = "B";

  {
    BackgroundQueue Q;
    Q.append({A, B});
    Q.work([&] { Q.stop(); });
    EXPECT_EQ("BA", Sequence) << "priority order";
  }
  Sequence.clear();
  {
    BackgroundQueue Q;
    Q.boost("A", 3);
    Q.append({A, B});
    Q.work([&] { Q.stop(); });
    EXPECT_EQ("AB", Sequence) << "A was boosted before enqueueing";
  }
  Sequence.clear();
  {
    BackgroundQueue Q;
    Q.append({A, B});
    Q.boost("A", 3);
    Q.work([&] { Q.stop(); });
    EXPECT_EQ("AB", Sequence) << "A was boosted after enqueueing";
  }
}

} // namespace clangd
} // namespace clang
