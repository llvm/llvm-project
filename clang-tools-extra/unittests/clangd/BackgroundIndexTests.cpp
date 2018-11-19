#include "SyncAPI.h"
#include "TestFS.h"
#include "index/Background.h"
#include "llvm/Support/ScopedPrinter.h"
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
  llvm::StringMap<std::string> Storage;
  size_t CacheHits = 0;
  MemoryShardStorage MSS(Storage, CacheHits);
  BackgroundIndex Idx(Context::empty(), "", FS, /*URISchemes=*/{"unittest"},
                      [&](llvm::StringRef) { return &MSS; });

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

TEST(BackgroundIndexTest, ShardStorageWriteTest) {
  MockFSProvider FS;
  FS.Files[testPath("root/A.h")] = R"cpp(
      void common();
      void f_b();
      class A_CC {};
      )cpp";
  std::string A_CC = "#include \"A.h\"\nvoid g() { (void)common; }";
  FS.Files[testPath("root/A.cc")] = A_CC;
  auto Digest = llvm::SHA1::hash(
      {reinterpret_cast<const uint8_t *>(A_CC.data()), A_CC.size()});

  llvm::StringMap<std::string> Storage;
  size_t CacheHits = 0;
  MemoryShardStorage MSS(Storage, CacheHits);

  tooling::CompileCommand Cmd;
  Cmd.Filename = testPath("root/A.cc");
  Cmd.Directory = testPath("root");
  Cmd.CommandLine = {"clang++", testPath("root/A.cc")};
  // Check nothing is loaded from Storage, but A.cc and A.h has been stored.
  {
    BackgroundIndex Idx(Context::empty(), "", FS, /*URISchemes=*/{"unittest"},
                        [&](llvm::StringRef) { return &MSS; });
    Idx.enqueue(testPath("root"), Cmd);
    Idx.blockUntilIdleForTest();
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
  EXPECT_EQ(*ShardSource->Digest, Digest);
}

} // namespace clangd
} // namespace clang
