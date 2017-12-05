//===-- ClangdTests.cpp - Clangd unit tests ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ClangdLSPServer.h"
#include "ClangdServer.h"
#include "Logger.h"
#include "TestFS.h"
#include "clang/Config/config.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace clang {
namespace clangd {
namespace {

// Don't wait for async ops in clangd test more than that to avoid blocking
// indefinitely in case of bugs.
static const std::chrono::seconds DefaultFutureTimeout =
    std::chrono::seconds(10);

static bool diagsContainErrors(ArrayRef<DiagWithFixIts> Diagnostics) {
  for (const auto &DiagAndFixIts : Diagnostics) {
    // FIXME: severities returned by clangd should have a descriptive
    // diagnostic severity enum
    const int ErrorSeverity = 1;
    if (DiagAndFixIts.Diag.severity == ErrorSeverity)
      return true;
  }
  return false;
}

class ErrorCheckingDiagConsumer : public DiagnosticsConsumer {
public:
  void
  onDiagnosticsReady(PathRef File,
                     Tagged<std::vector<DiagWithFixIts>> Diagnostics) override {
    bool HadError = diagsContainErrors(Diagnostics.Value);

    std::lock_guard<std::mutex> Lock(Mutex);
    HadErrorInLastDiags = HadError;
    LastVFSTag = Diagnostics.Tag;
  }

  bool hadErrorInLastDiags() {
    std::lock_guard<std::mutex> Lock(Mutex);
    return HadErrorInLastDiags;
  }

  VFSTag lastVFSTag() { return LastVFSTag; }

private:
  std::mutex Mutex;
  bool HadErrorInLastDiags = false;
  VFSTag LastVFSTag = VFSTag();
};

class ConstantFSProvider : public FileSystemProvider {
public:
  ConstantFSProvider(IntrusiveRefCntPtr<vfs::FileSystem> FS,
                     VFSTag Tag = VFSTag())
      : FS(std::move(FS)), Tag(std::move(Tag)) {}

  Tagged<IntrusiveRefCntPtr<vfs::FileSystem>>
  getTaggedFileSystem(PathRef File) override {
    return make_tagged(FS, Tag);
  }

private:
  IntrusiveRefCntPtr<vfs::FileSystem> FS;
  VFSTag Tag;
};

/// Replaces all patterns of the form 0x123abc with spaces
std::string replacePtrsInDump(std::string const &Dump) {
  llvm::Regex RE("0x[0-9a-fA-F]+");
  llvm::SmallVector<StringRef, 1> Matches;
  llvm::StringRef Pending = Dump;

  std::string Result;
  while (RE.match(Pending, &Matches)) {
    assert(Matches.size() == 1 && "Exactly one match expected");
    auto MatchPos = Matches[0].data() - Pending.data();

    Result += Pending.take_front(MatchPos);
    Pending = Pending.drop_front(MatchPos + Matches[0].size());
  }
  Result += Pending;

  return Result;
}

std::string dumpASTWithoutMemoryLocs(ClangdServer &Server, PathRef File) {
  auto DumpWithMemLocs = Server.dumpAST(File);
  return replacePtrsInDump(DumpWithMemLocs);
}

} // namespace

class ClangdVFSTest : public ::testing::Test {
protected:
  std::string parseSourceAndDumpAST(
      PathRef SourceFileRelPath, StringRef SourceContents,
      std::vector<std::pair<PathRef, StringRef>> ExtraFiles = {},
      bool ExpectErrors = false) {
    MockFSProvider FS;
    ErrorCheckingDiagConsumer DiagConsumer;
    MockCompilationDatabase CDB;
    ClangdServer Server(CDB, DiagConsumer, FS, getDefaultAsyncThreadsCount(),
                        /*StorePreamblesInMemory=*/true,
                        clangd::CodeCompleteOptions(),
                        EmptyLogger::getInstance());
    for (const auto &FileWithContents : ExtraFiles)
      FS.Files[getVirtualTestFilePath(FileWithContents.first)] =
          FileWithContents.second;

    auto SourceFilename = getVirtualTestFilePath(SourceFileRelPath);

    FS.ExpectedFile = SourceFilename;

    // Have to sync reparses because requests are processed on the calling
    // thread.
    auto AddDocFuture = Server.addDocument(SourceFilename, SourceContents);

    auto Result = dumpASTWithoutMemoryLocs(Server, SourceFilename);

    // Wait for reparse to finish before checking for errors.
    EXPECT_EQ(AddDocFuture.wait_for(DefaultFutureTimeout),
              std::future_status::ready);
    EXPECT_EQ(ExpectErrors, DiagConsumer.hadErrorInLastDiags());
    return Result;
  }
};

TEST_F(ClangdVFSTest, Parse) {
  // FIXME: figure out a stable format for AST dumps, so that we can check the
  // output of the dump itself is equal to the expected one, not just that it's
  // different.
  auto Empty = parseSourceAndDumpAST("foo.cpp", "", {});
  auto OneDecl = parseSourceAndDumpAST("foo.cpp", "int a;", {});
  auto SomeDecls = parseSourceAndDumpAST("foo.cpp", "int a; int b; int c;", {});
  EXPECT_NE(Empty, OneDecl);
  EXPECT_NE(Empty, SomeDecls);
  EXPECT_NE(SomeDecls, OneDecl);

  auto Empty2 = parseSourceAndDumpAST("foo.cpp", "");
  auto OneDecl2 = parseSourceAndDumpAST("foo.cpp", "int a;");
  auto SomeDecls2 = parseSourceAndDumpAST("foo.cpp", "int a; int b; int c;");
  EXPECT_EQ(Empty, Empty2);
  EXPECT_EQ(OneDecl, OneDecl2);
  EXPECT_EQ(SomeDecls, SomeDecls2);
}

TEST_F(ClangdVFSTest, ParseWithHeader) {
  parseSourceAndDumpAST("foo.cpp", "#include \"foo.h\"", {},
                        /*ExpectErrors=*/true);
  parseSourceAndDumpAST("foo.cpp", "#include \"foo.h\"", {{"foo.h", ""}},
                        /*ExpectErrors=*/false);

  const auto SourceContents = R"cpp(
#include "foo.h"
int b = a;
)cpp";
  parseSourceAndDumpAST("foo.cpp", SourceContents, {{"foo.h", ""}},
                        /*ExpectErrors=*/true);
  parseSourceAndDumpAST("foo.cpp", SourceContents, {{"foo.h", "int a;"}},
                        /*ExpectErrors=*/false);
}

TEST_F(ClangdVFSTest, Reparse) {
  MockFSProvider FS;
  ErrorCheckingDiagConsumer DiagConsumer;
  MockCompilationDatabase CDB;
  ClangdServer Server(CDB, DiagConsumer, FS, getDefaultAsyncThreadsCount(),
                      /*StorePreamblesInMemory=*/true,
                      clangd::CodeCompleteOptions(),
                      EmptyLogger::getInstance());

  const auto SourceContents = R"cpp(
#include "foo.h"
int b = a;
)cpp";

  auto FooCpp = getVirtualTestFilePath("foo.cpp");
  auto FooH = getVirtualTestFilePath("foo.h");

  FS.Files[FooH] = "int a;";
  FS.Files[FooCpp] = SourceContents;
  FS.ExpectedFile = FooCpp;

  // To sync reparses before checking for errors.
  std::future<void> ParseFuture;

  ParseFuture = Server.addDocument(FooCpp, SourceContents);
  auto DumpParse1 = dumpASTWithoutMemoryLocs(Server, FooCpp);
  ASSERT_EQ(ParseFuture.wait_for(DefaultFutureTimeout),
            std::future_status::ready);
  EXPECT_FALSE(DiagConsumer.hadErrorInLastDiags());

  ParseFuture = Server.addDocument(FooCpp, "");
  auto DumpParseEmpty = dumpASTWithoutMemoryLocs(Server, FooCpp);
  ASSERT_EQ(ParseFuture.wait_for(DefaultFutureTimeout),
            std::future_status::ready);
  EXPECT_FALSE(DiagConsumer.hadErrorInLastDiags());

  ParseFuture = Server.addDocument(FooCpp, SourceContents);
  auto DumpParse2 = dumpASTWithoutMemoryLocs(Server, FooCpp);
  ASSERT_EQ(ParseFuture.wait_for(DefaultFutureTimeout),
            std::future_status::ready);
  EXPECT_FALSE(DiagConsumer.hadErrorInLastDiags());

  EXPECT_EQ(DumpParse1, DumpParse2);
  EXPECT_NE(DumpParse1, DumpParseEmpty);
}

TEST_F(ClangdVFSTest, ReparseOnHeaderChange) {
  MockFSProvider FS;
  ErrorCheckingDiagConsumer DiagConsumer;
  MockCompilationDatabase CDB;

  ClangdServer Server(CDB, DiagConsumer, FS, getDefaultAsyncThreadsCount(),
                      /*StorePreamblesInMemory=*/true,
                      clangd::CodeCompleteOptions(),
                      EmptyLogger::getInstance());

  const auto SourceContents = R"cpp(
#include "foo.h"
int b = a;
)cpp";

  auto FooCpp = getVirtualTestFilePath("foo.cpp");
  auto FooH = getVirtualTestFilePath("foo.h");

  FS.Files[FooH] = "int a;";
  FS.Files[FooCpp] = SourceContents;
  FS.ExpectedFile = FooCpp;

  // To sync reparses before checking for errors.
  std::future<void> ParseFuture;

  ParseFuture = Server.addDocument(FooCpp, SourceContents);
  auto DumpParse1 = dumpASTWithoutMemoryLocs(Server, FooCpp);
  ASSERT_EQ(ParseFuture.wait_for(DefaultFutureTimeout),
            std::future_status::ready);
  EXPECT_FALSE(DiagConsumer.hadErrorInLastDiags());

  FS.Files[FooH] = "";
  ParseFuture = Server.forceReparse(FooCpp);
  auto DumpParseDifferent = dumpASTWithoutMemoryLocs(Server, FooCpp);
  ASSERT_EQ(ParseFuture.wait_for(DefaultFutureTimeout),
            std::future_status::ready);
  EXPECT_TRUE(DiagConsumer.hadErrorInLastDiags());

  FS.Files[FooH] = "int a;";
  ParseFuture = Server.forceReparse(FooCpp);
  auto DumpParse2 = dumpASTWithoutMemoryLocs(Server, FooCpp);
  EXPECT_EQ(ParseFuture.wait_for(DefaultFutureTimeout),
            std::future_status::ready);
  EXPECT_FALSE(DiagConsumer.hadErrorInLastDiags());

  EXPECT_EQ(DumpParse1, DumpParse2);
  EXPECT_NE(DumpParse1, DumpParseDifferent);
}

TEST_F(ClangdVFSTest, CheckVersions) {
  MockFSProvider FS;
  ErrorCheckingDiagConsumer DiagConsumer;
  MockCompilationDatabase CDB;
  // Run ClangdServer synchronously.
  ClangdServer Server(CDB, DiagConsumer, FS,
                      /*AsyncThreadsCount=*/0,
                      /*StorePreamblesInMemory=*/true,
                      clangd::CodeCompleteOptions(),
                      EmptyLogger::getInstance());

  auto FooCpp = getVirtualTestFilePath("foo.cpp");
  const auto SourceContents = "int a;";
  FS.Files[FooCpp] = SourceContents;
  FS.ExpectedFile = FooCpp;

  // No need to sync reparses, because requests are processed on the calling
  // thread.
  FS.Tag = "123";
  Server.addDocument(FooCpp, SourceContents);
  EXPECT_EQ(Server.codeComplete(FooCpp, Position{0, 0}).get().Tag, FS.Tag);
  EXPECT_EQ(DiagConsumer.lastVFSTag(), FS.Tag);

  FS.Tag = "321";
  Server.addDocument(FooCpp, SourceContents);
  EXPECT_EQ(DiagConsumer.lastVFSTag(), FS.Tag);
  EXPECT_EQ(Server.codeComplete(FooCpp, Position{0, 0}).get().Tag, FS.Tag);
}

// Only enable this test on Unix
#ifdef LLVM_ON_UNIX
TEST_F(ClangdVFSTest, SearchLibDir) {
  // Checks that searches for GCC installation is done through vfs.
  MockFSProvider FS;
  ErrorCheckingDiagConsumer DiagConsumer;
  MockCompilationDatabase CDB;
  CDB.ExtraClangFlags.insert(CDB.ExtraClangFlags.end(),
                             {"-xc++", "-target", "x86_64-linux-unknown",
                              "-m64", "--gcc-toolchain=/randomusr",
                              "-stdlib=libstdc++"});
  // Run ClangdServer synchronously.
  ClangdServer Server(CDB, DiagConsumer, FS,
                      /*AsyncThreadsCount=*/0,
                      /*StorePreamblesInMemory=*/true,
                      clangd::CodeCompleteOptions(),
                      EmptyLogger::getInstance());

  // Just a random gcc version string
  SmallString<8> Version("4.9.3");

  // A lib dir for gcc installation
  SmallString<64> LibDir("/randomusr/lib/gcc/x86_64-linux-gnu");
  llvm::sys::path::append(LibDir, Version);

  // Put crtbegin.o into LibDir/64 to trick clang into thinking there's a gcc
  // installation there.
  SmallString<64> DummyLibFile;
  llvm::sys::path::append(DummyLibFile, LibDir, "64", "crtbegin.o");
  FS.Files[DummyLibFile] = "";

  SmallString<64> IncludeDir("/randomusr/include/c++");
  llvm::sys::path::append(IncludeDir, Version);

  SmallString<64> StringPath;
  llvm::sys::path::append(StringPath, IncludeDir, "string");
  FS.Files[StringPath] = "class mock_string {};";

  auto FooCpp = getVirtualTestFilePath("foo.cpp");
  const auto SourceContents = R"cpp(
#include <string>
mock_string x;
)cpp";
  FS.Files[FooCpp] = SourceContents;

  // No need to sync reparses, because requests are processed on the calling
  // thread.
  Server.addDocument(FooCpp, SourceContents);
  EXPECT_FALSE(DiagConsumer.hadErrorInLastDiags());

  const auto SourceContentsWithError = R"cpp(
#include <string>
std::string x;
)cpp";
  Server.addDocument(FooCpp, SourceContentsWithError);
  EXPECT_TRUE(DiagConsumer.hadErrorInLastDiags());
}
#endif // LLVM_ON_UNIX

TEST_F(ClangdVFSTest, ForceReparseCompileCommand) {
  MockFSProvider FS;
  ErrorCheckingDiagConsumer DiagConsumer;
  MockCompilationDatabase CDB;
  ClangdServer Server(CDB, DiagConsumer, FS,
                      /*AsyncThreadsCount=*/0,
                      /*StorePreamblesInMemory=*/true,
                      clangd::CodeCompleteOptions(),
                      EmptyLogger::getInstance());
  // No need to sync reparses, because reparses are performed on the calling
  // thread to true.

  auto FooCpp = getVirtualTestFilePath("foo.cpp");
  const auto SourceContents1 = R"cpp(
template <class T>
struct foo { T x; };
)cpp";
  const auto SourceContents2 = R"cpp(
template <class T>
struct bar { T x; };
)cpp";

  FS.Files[FooCpp] = "";
  FS.ExpectedFile = FooCpp;

  // First parse files in C mode and check they produce errors.
  CDB.ExtraClangFlags = {"-xc"};
  Server.addDocument(FooCpp, SourceContents1);
  EXPECT_TRUE(DiagConsumer.hadErrorInLastDiags());
  Server.addDocument(FooCpp, SourceContents2);
  EXPECT_TRUE(DiagConsumer.hadErrorInLastDiags());

  // Now switch to C++ mode.
  CDB.ExtraClangFlags = {"-xc++"};
  // Currently, addDocument never checks if CompileCommand has changed, so we
  // expect to see the errors.
  Server.addDocument(FooCpp, SourceContents1);
  EXPECT_TRUE(DiagConsumer.hadErrorInLastDiags());
  Server.addDocument(FooCpp, SourceContents2);
  EXPECT_TRUE(DiagConsumer.hadErrorInLastDiags());
  // But forceReparse should reparse the file with proper flags.
  Server.forceReparse(FooCpp);
  EXPECT_FALSE(DiagConsumer.hadErrorInLastDiags());
  // Subsequent addDocument calls should finish without errors too.
  Server.addDocument(FooCpp, SourceContents1);
  EXPECT_FALSE(DiagConsumer.hadErrorInLastDiags());
  Server.addDocument(FooCpp, SourceContents2);
  EXPECT_FALSE(DiagConsumer.hadErrorInLastDiags());
}

class ClangdThreadingTest : public ClangdVFSTest {};

TEST_F(ClangdThreadingTest, StressTest) {
  // Without 'static' clang gives an error for a usage inside TestDiagConsumer.
  static const unsigned FilesCount = 5;
  const unsigned RequestsCount = 500;
  // Blocking requests wait for the parsing to complete, they slow down the test
  // dramatically, so they are issued rarely. Each
  // BlockingRequestInterval-request will be a blocking one.
  const unsigned BlockingRequestInterval = 40;

  const auto SourceContentsWithoutErrors = R"cpp(
int a;
int b;
int c;
int d;
)cpp";

  const auto SourceContentsWithErrors = R"cpp(
int a = x;
int b;
int c;
int d;
)cpp";

  // Giving invalid line and column number should not crash ClangdServer, but
  // just to make sure we're sometimes hitting the bounds inside the file we
  // limit the intervals of line and column number that are generated.
  unsigned MaxLineForFileRequests = 7;
  unsigned MaxColumnForFileRequests = 10;

  std::vector<SmallString<32>> FilePaths;
  FilePaths.reserve(FilesCount);
  for (unsigned I = 0; I < FilesCount; ++I)
    FilePaths.push_back(getVirtualTestFilePath(std::string("Foo") +
                                               std::to_string(I) + ".cpp"));
  // Mark all of those files as existing.
  llvm::StringMap<std::string> FileContents;
  for (auto &&FilePath : FilePaths)
    FileContents[FilePath] = "";

  ConstantFSProvider FS(buildTestFS(FileContents));

  struct FileStat {
    unsigned HitsWithoutErrors = 0;
    unsigned HitsWithErrors = 0;
    bool HadErrorsInLastDiags = false;
  };

  class TestDiagConsumer : public DiagnosticsConsumer {
  public:
    TestDiagConsumer() : Stats(FilesCount, FileStat()) {}

    void onDiagnosticsReady(
        PathRef File,
        Tagged<std::vector<DiagWithFixIts>> Diagnostics) override {
      StringRef FileIndexStr = llvm::sys::path::stem(File);
      ASSERT_TRUE(FileIndexStr.consume_front("Foo"));

      unsigned long FileIndex = std::stoul(FileIndexStr.str());

      bool HadError = diagsContainErrors(Diagnostics.Value);

      std::lock_guard<std::mutex> Lock(Mutex);
      if (HadError)
        Stats[FileIndex].HitsWithErrors++;
      else
        Stats[FileIndex].HitsWithoutErrors++;
      Stats[FileIndex].HadErrorsInLastDiags = HadError;
    }

    std::vector<FileStat> takeFileStats() {
      std::lock_guard<std::mutex> Lock(Mutex);
      return std::move(Stats);
    }

  private:
    std::mutex Mutex;
    std::vector<FileStat> Stats;
  };

  struct RequestStats {
    unsigned RequestsWithoutErrors = 0;
    unsigned RequestsWithErrors = 0;
    bool LastContentsHadErrors = false;
    bool FileIsRemoved = true;
    std::future<void> LastRequestFuture;
  };

  std::vector<RequestStats> ReqStats;
  ReqStats.reserve(FilesCount);
  for (unsigned FileIndex = 0; FileIndex < FilesCount; ++FileIndex)
    ReqStats.emplace_back();

  TestDiagConsumer DiagConsumer;
  {
    MockCompilationDatabase CDB;
    ClangdServer Server(CDB, DiagConsumer, FS, getDefaultAsyncThreadsCount(),
                        /*StorePreamblesInMemory=*/true,
                        clangd::CodeCompleteOptions(),
                        EmptyLogger::getInstance());

    // Prepare some random distributions for the test.
    std::random_device RandGen;

    std::uniform_int_distribution<unsigned> FileIndexDist(0, FilesCount - 1);
    // Pass a text that contains compiler errors to addDocument in about 20% of
    // all requests.
    std::bernoulli_distribution ShouldHaveErrorsDist(0.2);
    // Line and Column numbers for requests that need them.
    std::uniform_int_distribution<int> LineDist(0, MaxLineForFileRequests);
    std::uniform_int_distribution<int> ColumnDist(0, MaxColumnForFileRequests);

    // Some helpers.
    auto UpdateStatsOnAddDocument = [&](unsigned FileIndex, bool HadErrors,
                                        std::future<void> Future) {
      auto &Stats = ReqStats[FileIndex];

      if (HadErrors)
        ++Stats.RequestsWithErrors;
      else
        ++Stats.RequestsWithoutErrors;
      Stats.LastContentsHadErrors = HadErrors;
      Stats.FileIsRemoved = false;
      Stats.LastRequestFuture = std::move(Future);
    };

    auto UpdateStatsOnRemoveDocument = [&](unsigned FileIndex,
                                           std::future<void> Future) {
      auto &Stats = ReqStats[FileIndex];

      Stats.FileIsRemoved = true;
      Stats.LastRequestFuture = std::move(Future);
    };

    auto UpdateStatsOnForceReparse = [&](unsigned FileIndex,
                                         std::future<void> Future) {
      auto &Stats = ReqStats[FileIndex];

      Stats.LastRequestFuture = std::move(Future);
      if (Stats.LastContentsHadErrors)
        ++Stats.RequestsWithErrors;
      else
        ++Stats.RequestsWithoutErrors;
    };

    auto AddDocument = [&](unsigned FileIndex) {
      bool ShouldHaveErrors = ShouldHaveErrorsDist(RandGen);
      auto Future = Server.addDocument(
          FilePaths[FileIndex], ShouldHaveErrors ? SourceContentsWithErrors
                                                 : SourceContentsWithoutErrors);
      UpdateStatsOnAddDocument(FileIndex, ShouldHaveErrors, std::move(Future));
    };

    // Various requests that we would randomly run.
    auto AddDocumentRequest = [&]() {
      unsigned FileIndex = FileIndexDist(RandGen);
      AddDocument(FileIndex);
    };

    auto ForceReparseRequest = [&]() {
      unsigned FileIndex = FileIndexDist(RandGen);
      // Make sure we don't violate the ClangdServer's contract.
      if (ReqStats[FileIndex].FileIsRemoved)
        AddDocument(FileIndex);

      auto Future = Server.forceReparse(FilePaths[FileIndex]);
      UpdateStatsOnForceReparse(FileIndex, std::move(Future));
    };

    auto RemoveDocumentRequest = [&]() {
      unsigned FileIndex = FileIndexDist(RandGen);
      // Make sure we don't violate the ClangdServer's contract.
      if (ReqStats[FileIndex].FileIsRemoved)
        AddDocument(FileIndex);

      auto Future = Server.removeDocument(FilePaths[FileIndex]);
      UpdateStatsOnRemoveDocument(FileIndex, std::move(Future));
    };

    auto CodeCompletionRequest = [&]() {
      unsigned FileIndex = FileIndexDist(RandGen);
      // Make sure we don't violate the ClangdServer's contract.
      if (ReqStats[FileIndex].FileIsRemoved)
        AddDocument(FileIndex);

      Position Pos{LineDist(RandGen), ColumnDist(RandGen)};
      // FIXME(ibiryukov): Also test async completion requests.
      // Simply putting CodeCompletion into async requests now would make
      // tests slow, since there's no way to cancel previous completion
      // requests as opposed to AddDocument/RemoveDocument, which are implicitly
      // cancelled by any subsequent AddDocument/RemoveDocument request to the
      // same file.
      Server.codeComplete(FilePaths[FileIndex], Pos).wait();
    };

    auto FindDefinitionsRequest = [&]() {
      unsigned FileIndex = FileIndexDist(RandGen);
      // Make sure we don't violate the ClangdServer's contract.
      if (ReqStats[FileIndex].FileIsRemoved)
        AddDocument(FileIndex);

      Position Pos{LineDist(RandGen), ColumnDist(RandGen)};
      ASSERT_TRUE(!!Server.findDefinitions(FilePaths[FileIndex], Pos));
    };

    std::vector<std::function<void()>> AsyncRequests = {
        AddDocumentRequest, ForceReparseRequest, RemoveDocumentRequest};
    std::vector<std::function<void()>> BlockingRequests = {
        CodeCompletionRequest, FindDefinitionsRequest};

    // Bash requests to ClangdServer in a loop.
    std::uniform_int_distribution<int> AsyncRequestIndexDist(
        0, AsyncRequests.size() - 1);
    std::uniform_int_distribution<int> BlockingRequestIndexDist(
        0, BlockingRequests.size() - 1);
    for (unsigned I = 1; I <= RequestsCount; ++I) {
      if (I % BlockingRequestInterval != 0) {
        // Issue an async request most of the time. It should be fast.
        unsigned RequestIndex = AsyncRequestIndexDist(RandGen);
        AsyncRequests[RequestIndex]();
      } else {
        // Issue a blocking request once in a while.
        auto RequestIndex = BlockingRequestIndexDist(RandGen);
        BlockingRequests[RequestIndex]();
      }
    }

    // Wait for last requests to finish.
    for (auto &ReqStat : ReqStats) {
      if (!ReqStat.LastRequestFuture.valid())
        continue; // We never ran any requests for this file.

      // Future should be ready much earlier than in 5 seconds, the timeout is
      // there to check we won't wait indefinitely.
      ASSERT_EQ(ReqStat.LastRequestFuture.wait_for(std::chrono::seconds(5)),
                std::future_status::ready);
    }
  } // Wait for ClangdServer to shutdown before proceeding.

  // Check some invariants about the state of the program.
  std::vector<FileStat> Stats = DiagConsumer.takeFileStats();
  for (unsigned I = 0; I < FilesCount; ++I) {
    if (!ReqStats[I].FileIsRemoved) {
      ASSERT_EQ(Stats[I].HadErrorsInLastDiags,
                ReqStats[I].LastContentsHadErrors);
    }

    ASSERT_LE(Stats[I].HitsWithErrors, ReqStats[I].RequestsWithErrors);
    ASSERT_LE(Stats[I].HitsWithoutErrors, ReqStats[I].RequestsWithoutErrors);
  }
}

TEST_F(ClangdVFSTest, CheckSourceHeaderSwitch) {
  MockFSProvider FS;
  ErrorCheckingDiagConsumer DiagConsumer;
  MockCompilationDatabase CDB;

  ClangdServer Server(CDB, DiagConsumer, FS, getDefaultAsyncThreadsCount(),
                      /*StorePreamblesInMemory=*/true,
                      clangd::CodeCompleteOptions(),
                      EmptyLogger::getInstance());

  auto SourceContents = R"cpp(
  #include "foo.h"
  int b = a;
  )cpp";

  auto FooCpp = getVirtualTestFilePath("foo.cpp");
  auto FooH = getVirtualTestFilePath("foo.h");
  auto Invalid = getVirtualTestFilePath("main.cpp");

  FS.Files[FooCpp] = SourceContents;
  FS.Files[FooH] = "int a;";
  FS.Files[Invalid] = "int main() { \n return 0; \n }";

  llvm::Optional<Path> PathResult = Server.switchSourceHeader(FooCpp);
  EXPECT_TRUE(PathResult.hasValue());
  ASSERT_EQ(PathResult.getValue(), FooH);

  PathResult = Server.switchSourceHeader(FooH);
  EXPECT_TRUE(PathResult.hasValue());
  ASSERT_EQ(PathResult.getValue(), FooCpp);

  SourceContents = R"c(
  #include "foo.HH"
  int b = a;
  )c";

  // Test with header file in capital letters and different extension, source
  // file with different extension
  auto FooC = getVirtualTestFilePath("bar.c");
  auto FooHH = getVirtualTestFilePath("bar.HH");

  FS.Files[FooC] = SourceContents;
  FS.Files[FooHH] = "int a;";

  PathResult = Server.switchSourceHeader(FooC);
  EXPECT_TRUE(PathResult.hasValue());
  ASSERT_EQ(PathResult.getValue(), FooHH);

  // Test with both capital letters
  auto Foo2C = getVirtualTestFilePath("foo2.C");
  auto Foo2HH = getVirtualTestFilePath("foo2.HH");
  FS.Files[Foo2C] = SourceContents;
  FS.Files[Foo2HH] = "int a;";

  PathResult = Server.switchSourceHeader(Foo2C);
  EXPECT_TRUE(PathResult.hasValue());
  ASSERT_EQ(PathResult.getValue(), Foo2HH);

  // Test with source file as capital letter and .hxx header file
  auto Foo3C = getVirtualTestFilePath("foo3.C");
  auto Foo3HXX = getVirtualTestFilePath("foo3.hxx");

  SourceContents = R"c(
  #include "foo3.hxx"
  int b = a;
  )c";

  FS.Files[Foo3C] = SourceContents;
  FS.Files[Foo3HXX] = "int a;";

  PathResult = Server.switchSourceHeader(Foo3C);
  EXPECT_TRUE(PathResult.hasValue());
  ASSERT_EQ(PathResult.getValue(), Foo3HXX);

  // Test if asking for a corresponding file that doesn't exist returns an empty
  // string.
  PathResult = Server.switchSourceHeader(Invalid);
  EXPECT_FALSE(PathResult.hasValue());
}

TEST_F(ClangdThreadingTest, NoConcurrentDiagnostics) {
  class NoConcurrentAccessDiagConsumer : public DiagnosticsConsumer {
  public:
    NoConcurrentAccessDiagConsumer(std::promise<void> StartSecondReparse)
        : StartSecondReparse(std::move(StartSecondReparse)) {}

    void onDiagnosticsReady(
        PathRef File,
        Tagged<std::vector<DiagWithFixIts>> Diagnostics) override {

      std::unique_lock<std::mutex> Lock(Mutex, std::try_to_lock_t());
      ASSERT_TRUE(Lock.owns_lock())
          << "Detected concurrent onDiagnosticsReady calls for the same file.";
      if (FirstRequest) {
        FirstRequest = false;
        StartSecondReparse.set_value();
        // Sleep long enough for the second request to be processed.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    }

  private:
    std::mutex Mutex;
    bool FirstRequest = true;
    std::promise<void> StartSecondReparse;
  };

  const auto SourceContentsWithoutErrors = R"cpp(
int a;
int b;
int c;
int d;
)cpp";

  const auto SourceContentsWithErrors = R"cpp(
int a = x;
int b;
int c;
int d;
)cpp";

  auto FooCpp = getVirtualTestFilePath("foo.cpp");
  llvm::StringMap<std::string> FileContents;
  FileContents[FooCpp] = "";
  ConstantFSProvider FS(buildTestFS(FileContents));

  std::promise<void> StartSecondReparsePromise;
  std::future<void> StartSecondReparse = StartSecondReparsePromise.get_future();

  NoConcurrentAccessDiagConsumer DiagConsumer(
      std::move(StartSecondReparsePromise));

  MockCompilationDatabase CDB;
  ClangdServer Server(CDB, DiagConsumer, FS, 4, /*StorePreamblesInMemory=*/true,
                      clangd::CodeCompleteOptions(),
                      EmptyLogger::getInstance());
  Server.addDocument(FooCpp, SourceContentsWithErrors);
  StartSecondReparse.wait();

  auto Future = Server.addDocument(FooCpp, SourceContentsWithoutErrors);
  Future.wait();
}

} // namespace clangd
} // namespace clang
