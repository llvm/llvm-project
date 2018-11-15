#include "SyncAPI.h"
#include "TestFS.h"
#include "index/Background.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using testing::_;
using testing::AllOf;
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

TEST(BackgroundIndexTest, IndexTwoFiles) {
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
  BackgroundIndex Idx(Context::empty(), "", FS, /*URISchemes=*/{"unittest"});

  tooling::CompileCommand Cmd;
  Cmd.Filename = testPath("root/A.cc");
  Cmd.Directory = testPath("root");
  Cmd.CommandLine = {"clang++", "-DA=1", testPath("root/A.cc")};
  Idx.enqueue(testPath("root"), Cmd);

  Idx.blockUntilIdleForTest();
  EXPECT_THAT(
      runFuzzyFind(Idx, ""),
      UnorderedElementsAre(Named("common"), Named("A_CC"),
                           AllOf(Named("f_b"), Declared(), Not(Defined()))));

  Cmd.Filename = testPath("root/B.cc");
  Cmd.CommandLine = {"clang++", Cmd.Filename};
  Idx.enqueue(testPath("root"), Cmd);

  Idx.blockUntilIdleForTest();
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

TEST(BackgroundIndexTest, ShardStorageTest) {
  class MemoryShardStorage : public BackgroundIndexStorage {
    mutable std::mutex StorageMu;
    llvm::StringMap<std::string> &Storage;
    size_t &CacheHits;

  public:
    MemoryShardStorage(llvm::StringMap<std::string> &Storage, size_t &CacheHits)
        : Storage(Storage), CacheHits(CacheHits) {}

    bool storeShard(llvm::StringRef ShardIdentifier,
                    IndexFileOut Shard) const override {
      std::lock_guard<std::mutex> Lock(StorageMu);
      std::string &str = Storage[ShardIdentifier];
      llvm::raw_string_ostream OS(str);
      OS << Shard;
      OS.flush();
      return true;
    }
    llvm::Expected<IndexFileIn>
    retrieveShard(llvm::StringRef ShardIdentifier) const override {
      std::lock_guard<std::mutex> Lock(StorageMu);
      if (Storage.find(ShardIdentifier) == Storage.end())
        return llvm::make_error<llvm::StringError>(
            "Shard not found.", llvm::inconvertibleErrorCode());
      auto IndexFile = readIndexFile(Storage[ShardIdentifier]);
      if (!IndexFile)
        return IndexFile;
      CacheHits++;
      return IndexFile;
    }
    bool initialize(llvm::StringRef Directory) { return true; }
  };
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
  BackgroundIndexStorage::setStorageFactory(
      [&Storage, &CacheHits](llvm::StringRef) {
        return std::make_shared<MemoryShardStorage>(Storage, CacheHits);
      });

  tooling::CompileCommand Cmd;
  Cmd.Filename = testPath("root/A.cc");
  Cmd.Directory = testPath("root");
  Cmd.CommandLine = {"clang++", testPath("root/A.cc")};
  // Check nothing is loaded from Storage, but A.cc and A.h has been stored.
  {
    BackgroundIndex Idx(Context::empty(), "", FS, /*URISchemes=*/{"unittest"});
    Idx.enqueue(testPath("root"), Cmd);
    Idx.blockUntilIdleForTest();
  }
  EXPECT_EQ(CacheHits, 0U);
  EXPECT_EQ(Storage.size(), 2U);
  EXPECT_NE(Storage.find(testPath("root/A.h")), Storage.end());
  EXPECT_NE(Storage.find(testPath("root/A.cc")), Storage.end());

  // Check A.cc has been loaded from cache.
  {
    BackgroundIndex Idx(Context::empty(), "", FS, /*URISchemes=*/{"unittest"});
    Idx.enqueue(testPath("root"), Cmd);
    Idx.blockUntilIdleForTest();
  }
  EXPECT_EQ(CacheHits, 1U);
  EXPECT_EQ(Storage.size(), 2U);
  EXPECT_NE(Storage.find(testPath("root/A.h")), Storage.end());
  EXPECT_NE(Storage.find(testPath("root/A.cc")), Storage.end());
}

} // namespace clangd
} // namespace clang
