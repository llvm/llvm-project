#include "SyncAPI.h"
#include "TestFS.h"
#include "index/Background.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/Threading.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <thread>

using testing::_;
using testing::AllOf;
using testing::ElementsAre;
using testing::Not;
using testing::UnorderedElementsAre;

using namespace llvm;
namespace clang {
namespace clangd {

MATCHER_P(Named, N, "") { return arg.Name == N; }
MATCHER(Declared, "") {
  return !StringRef(arg.CanonicalDeclaration.FileURI).empty();
}
MATCHER(Defined, "") { return !StringRef(arg.Definition.FileURI).empty(); }
MATCHER_P(FileURI, F, "") { return StringRef(arg.Location.FileURI) == F; }
testing::Matcher<const RefSlab &>
RefsAre(std::vector<testing::Matcher<Ref>> Matchers) {
  return ElementsAre(testing::Pair(_, UnorderedElementsAreArray(Matchers)));
}
// URI cannot be empty since it references keys in the IncludeGraph.
MATCHER(EmptyIncludeNode, "") {
  return !arg.IsTU && !arg.URI.empty() && arg.Digest == FileDigest{{0}} &&
         arg.DirectIncludes.empty();
}

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
    Storage[ShardIdentifier] = llvm::to_string(Shard);
    return llvm::Error::success();
  }
  std::unique_ptr<IndexFileIn>
  loadShard(llvm::StringRef ShardIdentifier) const override {
    std::lock_guard<std::mutex> Lock(StorageMu);
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
    return llvm::make_unique<IndexFileIn>(std::move(*IndexFile));
  }
};

class BackgroundIndexTest : public ::testing::Test {
protected:
  BackgroundIndexTest() { preventThreadStarvationInTests(); }
};

TEST_F(BackgroundIndexTest, NoCrashOnErrorFile) {
  MockFSProvider FS;
  FS.Files[testPath("root/A.cc")] = "error file";
  llvm::StringMap<std::string> Storage;
  size_t CacheHits = 0;
  MemoryShardStorage MSS(Storage, CacheHits);
  OverlayCDB CDB(/*Base=*/nullptr);
  BackgroundIndex Idx(Context::empty(), "", FS, CDB,
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
      })cpp";
  llvm::StringMap<std::string> Storage;
  size_t CacheHits = 0;
  MemoryShardStorage MSS(Storage, CacheHits);
  OverlayCDB CDB(/*Base=*/nullptr);
  BackgroundIndex Idx(Context::empty(), "", FS, CDB,
                      [&](llvm::StringRef) { return &MSS; });

  tooling::CompileCommand Cmd;
  Cmd.Filename = testPath("root/A.cc");
  Cmd.Directory = testPath("root");
  Cmd.CommandLine = {"clang++", "-DA=1", testPath("root/A.cc")};
  CDB.setCompileCommand(testPath("root/A.cc"), Cmd);

  ASSERT_TRUE(Idx.blockUntilIdleForTest());
  EXPECT_THAT(
      runFuzzyFind(Idx, ""),
      UnorderedElementsAre(Named("common"), Named("A_CC"),
                           AllOf(Named("f_b"), Declared(), Not(Defined()))));

  Cmd.Filename = testPath("root/B.cc");
  Cmd.CommandLine = {"clang++", Cmd.Filename};
  CDB.setCompileCommand(testPath("root/A.cc"), Cmd);

  ASSERT_TRUE(Idx.blockUntilIdleForTest());
  // B_CC is dropped as we don't collect symbols from A.h in this compilation.
  EXPECT_THAT(runFuzzyFind(Idx, ""),
              UnorderedElementsAre(Named("common"), Named("A_CC"),
                                   AllOf(Named("f_b"), Declared(), Defined())));

  auto Syms = runFuzzyFind(Idx, "common");
  EXPECT_THAT(Syms, UnorderedElementsAre(Named("common")));
  auto Common = *Syms.begin();
  EXPECT_THAT(getRefs(Idx, Common.ID),
              RefsAre({FileURI("unittest:///root/A.h"),
                       FileURI("unittest:///root/A.cc"),
                       FileURI("unittest:///root/B.cc")}));
}

TEST_F(BackgroundIndexTest, ShardStorageWriteTest) {
  MockFSProvider FS;
  FS.Files[testPath("root/A.h")] = R"cpp(
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
  // Check nothing is loaded from Storage, but A.cc and A.h has been stored.
  {
    OverlayCDB CDB(/*Base=*/nullptr);
    BackgroundIndex Idx(Context::empty(), "", FS, CDB,
                        [&](llvm::StringRef) { return &MSS; });
    CDB.setCompileCommand(testPath("root/A.cc"), Cmd);
    ASSERT_TRUE(Idx.blockUntilIdleForTest());
  }
  EXPECT_EQ(CacheHits, 0U);
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
  EXPECT_THAT(*ShardSource->Symbols, UnorderedElementsAre());
  EXPECT_THAT(*ShardSource->Refs, RefsAre({FileURI("unittest:///root/A.cc")}));
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
    BackgroundIndex Idx(Context::empty(), "", FS, CDB,
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

TEST_F(BackgroundIndexTest, PeriodicalIndex) {
  MockFSProvider FS;
  llvm::StringMap<std::string> Storage;
  size_t CacheHits = 0;
  MemoryShardStorage MSS(Storage, CacheHits);
  OverlayCDB CDB(/*Base=*/nullptr);
  BackgroundIndex Idx(
      Context::empty(), "", FS, CDB, [&](llvm::StringRef) { return &MSS; },
      /*BuildIndexPeriodMs=*/500);

  FS.Files[testPath("root/A.cc")] = "#include \"A.h\"";

  tooling::CompileCommand Cmd;
  FS.Files[testPath("root/A.h")] = "class X {};";
  Cmd.Filename = testPath("root/A.cc");
  Cmd.CommandLine = {"clang++", Cmd.Filename};
  CDB.setCompileCommand(testPath("root/A.cc"), Cmd);

  ASSERT_TRUE(Idx.blockUntilIdleForTest());
  EXPECT_THAT(runFuzzyFind(Idx, ""), ElementsAre());
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  EXPECT_THAT(runFuzzyFind(Idx, ""), ElementsAre(Named("X")));

  FS.Files[testPath("root/A.h")] = "class Y {};";
  FS.Files[testPath("root/A.cc")] += " "; // Force reindex the file.
  Cmd.CommandLine = {"clang++", "-DA=1", testPath("root/A.cc")};
  CDB.setCompileCommand(testPath("root/A.cc"), Cmd);

  ASSERT_TRUE(Idx.blockUntilIdleForTest());
  EXPECT_THAT(runFuzzyFind(Idx, ""), ElementsAre(Named("X")));
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  EXPECT_THAT(runFuzzyFind(Idx, ""), ElementsAre(Named("Y")));
}

} // namespace clangd
} // namespace clang
