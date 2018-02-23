//===-- ClangdTests.cpp - Clangd unit tests ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "ClangdLSPServer.h"
#include "ClangdServer.h"
#include "Matchers.h"
#include "SyncAPI.h"
#include "TestFS.h"
#include "URI.h"
#include "clang/Config/config.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"
#include "gmock/gmock.h"
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

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Gt;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

namespace {

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

/// For each file, record whether the last published diagnostics contained at
/// least one error.
class MultipleErrorCheckingDiagConsumer : public DiagnosticsConsumer {
public:
  void
  onDiagnosticsReady(PathRef File,
                     Tagged<std::vector<DiagWithFixIts>> Diagnostics) override {
    bool HadError = diagsContainErrors(Diagnostics.Value);

    std::lock_guard<std::mutex> Lock(Mutex);
    LastDiagsHadError[File] = HadError;
  }

  /// Exposes all files consumed by onDiagnosticsReady in an unspecified order.
  /// For each file, a bool value indicates whether the last diagnostics
  /// contained an error.
  std::vector<std::pair<Path, bool>> filesWithDiags() const {
    std::vector<std::pair<Path, bool>> Result;
    std::lock_guard<std::mutex> Lock(Mutex);

    for (const auto &it : LastDiagsHadError) {
      Result.emplace_back(it.first(), it.second);
    }

    return Result;
  }

  void clear() {
    std::lock_guard<std::mutex> Lock(Mutex);
    LastDiagsHadError.clear();
  }

private:
  mutable std::mutex Mutex;
  llvm::StringMap<bool> LastDiagsHadError;
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
  auto DumpWithMemLocs = runDumpAST(Server, File);
  return replacePtrsInDump(DumpWithMemLocs);
}

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
                        /*StorePreamblesInMemory=*/true);
    for (const auto &FileWithContents : ExtraFiles)
      FS.Files[testPath(FileWithContents.first)] = FileWithContents.second;

    auto SourceFilename = testPath(SourceFileRelPath);
    FS.ExpectedFile = SourceFilename;
    Server.addDocument(SourceFilename, SourceContents);
    auto Result = dumpASTWithoutMemoryLocs(Server, SourceFilename);
    EXPECT_TRUE(Server.blockUntilIdleForTest()) << "Waiting for diagnostics";
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
                      /*StorePreamblesInMemory=*/true);

  const auto SourceContents = R"cpp(
#include "foo.h"
int b = a;
)cpp";

  auto FooCpp = testPath("foo.cpp");

  FS.Files[testPath("foo.h")] = "int a;";
  FS.Files[FooCpp] = SourceContents;
  FS.ExpectedFile = FooCpp;

  Server.addDocument(FooCpp, SourceContents);
  auto DumpParse1 = dumpASTWithoutMemoryLocs(Server, FooCpp);
  ASSERT_TRUE(Server.blockUntilIdleForTest()) << "Waiting for diagnostics";
  EXPECT_FALSE(DiagConsumer.hadErrorInLastDiags());

  Server.addDocument(FooCpp, "");
  auto DumpParseEmpty = dumpASTWithoutMemoryLocs(Server, FooCpp);
  ASSERT_TRUE(Server.blockUntilIdleForTest()) << "Waiting for diagnostics";
  EXPECT_FALSE(DiagConsumer.hadErrorInLastDiags());

  Server.addDocument(FooCpp, SourceContents);
  auto DumpParse2 = dumpASTWithoutMemoryLocs(Server, FooCpp);
  ASSERT_TRUE(Server.blockUntilIdleForTest()) << "Waiting for diagnostics";
  EXPECT_FALSE(DiagConsumer.hadErrorInLastDiags());

  EXPECT_EQ(DumpParse1, DumpParse2);
  EXPECT_NE(DumpParse1, DumpParseEmpty);
}

TEST_F(ClangdVFSTest, ReparseOnHeaderChange) {
  MockFSProvider FS;
  ErrorCheckingDiagConsumer DiagConsumer;
  MockCompilationDatabase CDB;

  ClangdServer Server(CDB, DiagConsumer, FS, getDefaultAsyncThreadsCount(),
                      /*StorePreamblesInMemory=*/true);

  const auto SourceContents = R"cpp(
#include "foo.h"
int b = a;
)cpp";

  auto FooCpp = testPath("foo.cpp");
  auto FooH = testPath("foo.h");

  FS.Files[FooH] = "int a;";
  FS.Files[FooCpp] = SourceContents;
  FS.ExpectedFile = FooCpp;

  Server.addDocument(FooCpp, SourceContents);
  auto DumpParse1 = dumpASTWithoutMemoryLocs(Server, FooCpp);
  ASSERT_TRUE(Server.blockUntilIdleForTest()) << "Waiting for diagnostics";
  EXPECT_FALSE(DiagConsumer.hadErrorInLastDiags());

  FS.Files[FooH] = "";
  Server.forceReparse(FooCpp);
  auto DumpParseDifferent = dumpASTWithoutMemoryLocs(Server, FooCpp);
  ASSERT_TRUE(Server.blockUntilIdleForTest()) << "Waiting for diagnostics";
  EXPECT_TRUE(DiagConsumer.hadErrorInLastDiags());

  FS.Files[FooH] = "int a;";
  Server.forceReparse(FooCpp);
  auto DumpParse2 = dumpASTWithoutMemoryLocs(Server, FooCpp);
  ASSERT_TRUE(Server.blockUntilIdleForTest()) << "Waiting for diagnostics";
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
                      /*StorePreamblesInMemory=*/true);

  auto FooCpp = testPath("foo.cpp");
  const auto SourceContents = "int a;";
  FS.Files[FooCpp] = SourceContents;
  FS.ExpectedFile = FooCpp;

  // Use default completion options.
  clangd::CodeCompleteOptions CCOpts;

  // No need to sync reparses, because requests are processed on the calling
  // thread.
  FS.Tag = "123";
  Server.addDocument(FooCpp, SourceContents);
  EXPECT_EQ(runCodeComplete(Server, FooCpp, Position(), CCOpts).Tag, FS.Tag);
  EXPECT_EQ(DiagConsumer.lastVFSTag(), FS.Tag);

  FS.Tag = "321";
  Server.addDocument(FooCpp, SourceContents);
  EXPECT_EQ(DiagConsumer.lastVFSTag(), FS.Tag);
  EXPECT_EQ(runCodeComplete(Server, FooCpp, Position(), CCOpts).Tag, FS.Tag);
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
                      /*StorePreamblesInMemory=*/true);

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

  auto FooCpp = testPath("foo.cpp");
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
                      /*StorePreamblesInMemory=*/true);

  // No need to sync reparses, because reparses are performed on the calling
  // thread.
  auto FooCpp = testPath("foo.cpp");
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

TEST_F(ClangdVFSTest, ForceReparseCompileCommandDefines) {
  MockFSProvider FS;
  ErrorCheckingDiagConsumer DiagConsumer;
  MockCompilationDatabase CDB;
  ClangdServer Server(CDB, DiagConsumer, FS,
                      /*AsyncThreadsCount=*/0,
                      /*StorePreamblesInMemory=*/true);

  // No need to sync reparses, because reparses are performed on the calling
  // thread.
  auto FooCpp = testPath("foo.cpp");
  const auto SourceContents = R"cpp(
#ifdef WITH_ERROR
this
#endif

int main() { return 0; }
)cpp";
  FS.Files[FooCpp] = "";
  FS.ExpectedFile = FooCpp;

  // Parse with define, we expect to see the errors.
  CDB.ExtraClangFlags = {"-DWITH_ERROR"};
  Server.addDocument(FooCpp, SourceContents);
  EXPECT_TRUE(DiagConsumer.hadErrorInLastDiags());

  // Parse without the define, no errors should be produced.
  CDB.ExtraClangFlags = {};
  // Currently, addDocument never checks if CompileCommand has changed, so we
  // expect to see the errors.
  Server.addDocument(FooCpp, SourceContents);
  EXPECT_TRUE(DiagConsumer.hadErrorInLastDiags());
  // But forceReparse should reparse the file with proper flags.
  Server.forceReparse(FooCpp);
  EXPECT_FALSE(DiagConsumer.hadErrorInLastDiags());
  // Subsequent addDocument call should finish without errors too.
  Server.addDocument(FooCpp, SourceContents);
  EXPECT_FALSE(DiagConsumer.hadErrorInLastDiags());
}

// Test ClangdServer.reparseOpenedFiles.
TEST_F(ClangdVFSTest, ReparseOpenedFiles) {
  Annotations FooSource(R"cpp(
#ifdef MACRO
$one[[static void bob() {}]]
#else
$two[[static void bob() {}]]
#endif

int main () { bo^b (); return 0; }
)cpp");

  Annotations BarSource(R"cpp(
#ifdef MACRO
this is an error
#endif
)cpp");

  Annotations BazSource(R"cpp(
int hello;
)cpp");

  MockFSProvider FS;
  MockCompilationDatabase CDB;
  MultipleErrorCheckingDiagConsumer DiagConsumer;
  ClangdServer Server(CDB, DiagConsumer, FS,
                      /*AsyncThreadsCount=*/0,
                      /*StorePreamblesInMemory=*/true);

  auto FooCpp = testPath("foo.cpp");
  auto BarCpp = testPath("bar.cpp");
  auto BazCpp = testPath("baz.cpp");

  FS.Files[FooCpp] = "";
  FS.Files[BarCpp] = "";
  FS.Files[BazCpp] = "";

  CDB.ExtraClangFlags = {"-DMACRO=1"};
  Server.addDocument(FooCpp, FooSource.code());
  Server.addDocument(BarCpp, BarSource.code());
  Server.addDocument(BazCpp, BazSource.code());

  EXPECT_THAT(DiagConsumer.filesWithDiags(),
              UnorderedElementsAre(Pair(FooCpp, false), Pair(BarCpp, true),
                                   Pair(BazCpp, false)));

  auto Locations = runFindDefinitions(Server, FooCpp, FooSource.point());
  EXPECT_TRUE(bool(Locations));
  EXPECT_THAT(Locations->Value, ElementsAre(Location{URIForFile{FooCpp},
                                                     FooSource.range("one")}));

  // Undefine MACRO, close baz.cpp.
  CDB.ExtraClangFlags.clear();
  DiagConsumer.clear();
  Server.removeDocument(BazCpp);
  Server.reparseOpenedFiles();

  EXPECT_THAT(DiagConsumer.filesWithDiags(),
              UnorderedElementsAre(Pair(FooCpp, false), Pair(BarCpp, false)));

  Locations = runFindDefinitions(Server, FooCpp, FooSource.point());
  EXPECT_TRUE(bool(Locations));
  EXPECT_THAT(Locations->Value, ElementsAre(Location{URIForFile{FooCpp},
                                                     FooSource.range("two")}));
}

TEST_F(ClangdVFSTest, MemoryUsage) {
  MockFSProvider FS;
  ErrorCheckingDiagConsumer DiagConsumer;
  MockCompilationDatabase CDB;
  ClangdServer Server(CDB, DiagConsumer, FS,
                      /*AsyncThreadsCount=*/0,
                      /*StorePreamblesInMemory=*/true);

  // No need to sync reparses, because reparses are performed on the calling
  // thread.
  Path FooCpp = testPath("foo.cpp");
  const auto SourceContents = R"cpp(
struct Something {
  int method();
};
)cpp";
  Path BarCpp = testPath("bar.cpp");

  FS.Files[FooCpp] = "";
  FS.Files[BarCpp] = "";

  EXPECT_THAT(Server.getUsedBytesPerFile(), IsEmpty());

  Server.addDocument(FooCpp, SourceContents);
  Server.addDocument(BarCpp, SourceContents);

  EXPECT_THAT(Server.getUsedBytesPerFile(),
              UnorderedElementsAre(Pair(FooCpp, Gt(0u)), Pair(BarCpp, Gt(0u))));

  Server.removeDocument(FooCpp);
  EXPECT_THAT(Server.getUsedBytesPerFile(), ElementsAre(Pair(BarCpp, Gt(0u))));

  Server.removeDocument(BarCpp);
  EXPECT_THAT(Server.getUsedBytesPerFile(), IsEmpty());
}

TEST_F(ClangdVFSTest, InvalidCompileCommand) {
  MockFSProvider FS;
  ErrorCheckingDiagConsumer DiagConsumer;
  MockCompilationDatabase CDB;

  ClangdServer Server(CDB, DiagConsumer, FS,
                      /*AsyncThreadsCount=*/0,
                      /*StorePreamblesInMemory=*/true);

  auto FooCpp = testPath("foo.cpp");
  // clang cannot create CompilerInvocation if we pass two files in the
  // CompileCommand. We pass the file in ExtraFlags once and CDB adds another
  // one in getCompileCommand().
  CDB.ExtraClangFlags.push_back(FooCpp);

  // Clang can't parse command args in that case, but we shouldn't crash.
  Server.addDocument(FooCpp, "int main() {}");

  EXPECT_EQ(runDumpAST(Server, FooCpp), "<no-ast>");
  EXPECT_ERROR(runFindDefinitions(Server, FooCpp, Position()));
  EXPECT_ERROR(runFindDocumentHighlights(Server, FooCpp, Position()));
  EXPECT_ERROR(runRename(Server, FooCpp, Position(), "new_name"));
  // FIXME: codeComplete and signatureHelp should also return errors when they
  // can't parse the file.
  EXPECT_THAT(
      runCodeComplete(Server, FooCpp, Position(), clangd::CodeCompleteOptions())
          .Value.items,
      IsEmpty());
  auto SigHelp = runSignatureHelp(Server, FooCpp, Position());
  ASSERT_TRUE(bool(SigHelp)) << "signatureHelp returned an error";
  EXPECT_THAT(SigHelp->Value.signatures, IsEmpty());
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

  std::vector<std::string> FilePaths;
  MockFSProvider FS;
  for (unsigned I = 0; I < FilesCount; ++I) {
    std::string Name = std::string("Foo") + std::to_string(I) + ".cpp";
    FS.Files[Name] = "";
    FilePaths.push_back(testPath(Name));
  }

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
  };

  std::vector<RequestStats> ReqStats;
  ReqStats.reserve(FilesCount);
  for (unsigned FileIndex = 0; FileIndex < FilesCount; ++FileIndex)
    ReqStats.emplace_back();

  TestDiagConsumer DiagConsumer;
  {
    MockCompilationDatabase CDB;
    ClangdServer Server(CDB, DiagConsumer, FS, getDefaultAsyncThreadsCount(),
                        /*StorePreamblesInMemory=*/true);

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
    auto UpdateStatsOnAddDocument = [&](unsigned FileIndex, bool HadErrors) {
      auto &Stats = ReqStats[FileIndex];

      if (HadErrors)
        ++Stats.RequestsWithErrors;
      else
        ++Stats.RequestsWithoutErrors;
      Stats.LastContentsHadErrors = HadErrors;
      Stats.FileIsRemoved = false;
    };

    auto UpdateStatsOnRemoveDocument = [&](unsigned FileIndex) {
      auto &Stats = ReqStats[FileIndex];

      Stats.FileIsRemoved = true;
    };

    auto UpdateStatsOnForceReparse = [&](unsigned FileIndex) {
      auto &Stats = ReqStats[FileIndex];

      if (Stats.LastContentsHadErrors)
        ++Stats.RequestsWithErrors;
      else
        ++Stats.RequestsWithoutErrors;
    };

    auto AddDocument = [&](unsigned FileIndex) {
      bool ShouldHaveErrors = ShouldHaveErrorsDist(RandGen);
      Server.addDocument(FilePaths[FileIndex],
                         ShouldHaveErrors ? SourceContentsWithErrors
                                          : SourceContentsWithoutErrors);
      UpdateStatsOnAddDocument(FileIndex, ShouldHaveErrors);
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

      Server.forceReparse(FilePaths[FileIndex]);
      UpdateStatsOnForceReparse(FileIndex);
    };

    auto RemoveDocumentRequest = [&]() {
      unsigned FileIndex = FileIndexDist(RandGen);
      // Make sure we don't violate the ClangdServer's contract.
      if (ReqStats[FileIndex].FileIsRemoved)
        AddDocument(FileIndex);

      Server.removeDocument(FilePaths[FileIndex]);
      UpdateStatsOnRemoveDocument(FileIndex);
    };

    auto CodeCompletionRequest = [&]() {
      unsigned FileIndex = FileIndexDist(RandGen);
      // Make sure we don't violate the ClangdServer's contract.
      if (ReqStats[FileIndex].FileIsRemoved)
        AddDocument(FileIndex);

      Position Pos;
      Pos.line = LineDist(RandGen);
      Pos.character = ColumnDist(RandGen);
      // FIXME(ibiryukov): Also test async completion requests.
      // Simply putting CodeCompletion into async requests now would make
      // tests slow, since there's no way to cancel previous completion
      // requests as opposed to AddDocument/RemoveDocument, which are implicitly
      // cancelled by any subsequent AddDocument/RemoveDocument request to the
      // same file.
      runCodeComplete(Server, FilePaths[FileIndex], Pos,
                      clangd::CodeCompleteOptions());
    };

    auto FindDefinitionsRequest = [&]() {
      unsigned FileIndex = FileIndexDist(RandGen);
      // Make sure we don't violate the ClangdServer's contract.
      if (ReqStats[FileIndex].FileIsRemoved)
        AddDocument(FileIndex);

      Position Pos;
      Pos.line = LineDist(RandGen);
      Pos.character = ColumnDist(RandGen);

      ASSERT_TRUE(!!runFindDefinitions(Server, FilePaths[FileIndex], Pos));
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
                      /*StorePreamblesInMemory=*/true);

  auto SourceContents = R"cpp(
  #include "foo.h"
  int b = a;
  )cpp";

  auto FooCpp = testPath("foo.cpp");
  auto FooH = testPath("foo.h");
  auto Invalid = testPath("main.cpp");

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
  auto FooC = testPath("bar.c");
  auto FooHH = testPath("bar.HH");

  FS.Files[FooC] = SourceContents;
  FS.Files[FooHH] = "int a;";

  PathResult = Server.switchSourceHeader(FooC);
  EXPECT_TRUE(PathResult.hasValue());
  ASSERT_EQ(PathResult.getValue(), FooHH);

  // Test with both capital letters
  auto Foo2C = testPath("foo2.C");
  auto Foo2HH = testPath("foo2.HH");
  FS.Files[Foo2C] = SourceContents;
  FS.Files[Foo2HH] = "int a;";

  PathResult = Server.switchSourceHeader(Foo2C);
  EXPECT_TRUE(PathResult.hasValue());
  ASSERT_EQ(PathResult.getValue(), Foo2HH);

  // Test with source file as capital letter and .hxx header file
  auto Foo3C = testPath("foo3.C");
  auto Foo3HXX = testPath("foo3.hxx");

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
    std::atomic<int> Count = {0};

    NoConcurrentAccessDiagConsumer(std::promise<void> StartSecondReparse)
        : StartSecondReparse(std::move(StartSecondReparse)) {}

    void onDiagnosticsReady(PathRef,
                            Tagged<std::vector<DiagWithFixIts>>) override {
      ++Count;
      std::unique_lock<std::mutex> Lock(Mutex, std::try_to_lock_t());
      ASSERT_TRUE(Lock.owns_lock())
          << "Detected concurrent onDiagnosticsReady calls for the same file.";

      // If we started the second parse immediately, it might cancel the first.
      // So we don't allow it to start until the first has delivered diags...
      if (FirstRequest) {
        FirstRequest = false;
        StartSecondReparse.set_value();
        // ... but then we wait long enough that the callbacks would overlap.
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

  auto FooCpp = testPath("foo.cpp");
  MockFSProvider FS;
  FS.Files[FooCpp] = "";

  std::promise<void> StartSecondPromise;
  std::future<void> StartSecond = StartSecondPromise.get_future();

  NoConcurrentAccessDiagConsumer DiagConsumer(std::move(StartSecondPromise));
  MockCompilationDatabase CDB;
  ClangdServer Server(CDB, DiagConsumer, FS, /*AsyncThreadsCount=*/4,
                      /*StorePreamblesInMemory=*/true);
  Server.addDocument(FooCpp, SourceContentsWithErrors);
  StartSecond.wait();
  Server.addDocument(FooCpp, SourceContentsWithoutErrors);
  ASSERT_TRUE(Server.blockUntilIdleForTest()) << "Waiting for diagnostics";
  ASSERT_EQ(DiagConsumer.Count, 2); // Sanity check - we actually ran both?
}

TEST_F(ClangdVFSTest, InsertIncludes) {
  MockFSProvider FS;
  ErrorCheckingDiagConsumer DiagConsumer;
  MockCompilationDatabase CDB;
  ClangdServer Server(CDB, DiagConsumer, FS,
                      /*AsyncThreadsCount=*/0,
                      /*StorePreamblesInMemory=*/true);

  // No need to sync reparses, because reparses are performed on the calling
  // thread.
  auto FooCpp = testPath("foo.cpp");
  const auto Code = R"cpp(
#include "x.h"

void f() {}
)cpp";
  FS.Files[FooCpp] = Code;
  Server.addDocument(FooCpp, Code);

  auto Inserted = [&](llvm::StringRef Header, llvm::StringRef Expected) {
    auto Replaces = Server.insertInclude(FooCpp, Code, Header);
    EXPECT_TRUE(static_cast<bool>(Replaces));
    auto Changed = tooling::applyAllReplacements(Code, *Replaces);
    EXPECT_TRUE(static_cast<bool>(Changed));
    return llvm::StringRef(*Changed).contains(
        (llvm::Twine("#include ") + Expected + "").str());
  };

  EXPECT_TRUE(Inserted("\"y.h\"", "\"y.h\""));
  EXPECT_TRUE(Inserted("<string>", "<string>"));
}

} // namespace
} // namespace clangd
} // namespace clang
