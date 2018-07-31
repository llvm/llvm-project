//===-- TUSchedulerTests.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Context.h"
#include "TUScheduler.h"
#include "TestFS.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <utility>

namespace clang {
namespace clangd {

using ::testing::_;
using ::testing::Each;
using ::testing::AnyOf;
using ::testing::Pair;
using ::testing::Pointee;
using ::testing::UnorderedElementsAre;

void ignoreUpdate(llvm::Optional<std::vector<Diag>>) {}
void ignoreError(llvm::Error Err) {
  handleAllErrors(std::move(Err), [](const llvm::ErrorInfoBase &) {});
}

class TUSchedulerTests : public ::testing::Test {
protected:
  ParseInputs getInputs(PathRef File, std::string Contents) {
    return ParseInputs{*CDB.getCompileCommand(File),
                       buildTestFS(Files, Timestamps), std::move(Contents)};
  }

  llvm::StringMap<std::string> Files;
  llvm::StringMap<time_t> Timestamps;
  MockCompilationDatabase CDB;
};

TEST_F(TUSchedulerTests, MissingFiles) {
  TUScheduler S(getDefaultAsyncThreadsCount(),
                /*StorePreamblesInMemory=*/true,
                /*PreambleParsedCallback=*/nullptr,
                /*UpdateDebounce=*/std::chrono::steady_clock::duration::zero(),
                ASTRetentionPolicy());

  auto Added = testPath("added.cpp");
  Files[Added] = "";

  auto Missing = testPath("missing.cpp");
  Files[Missing] = "";

  S.update(Added, getInputs(Added, ""), WantDiagnostics::No, ignoreUpdate);

  // Assert each operation for missing file is an error (even if it's available
  // in VFS).
  S.runWithAST("", Missing, [&](llvm::Expected<InputsAndAST> AST) {
    ASSERT_FALSE(bool(AST));
    ignoreError(AST.takeError());
  });
  S.runWithPreamble("", Missing,
                    [&](llvm::Expected<InputsAndPreamble> Preamble) {
                      ASSERT_FALSE(bool(Preamble));
                      ignoreError(Preamble.takeError());
                    });
  // remove() shouldn't crash on missing files.
  S.remove(Missing);

  // Assert there aren't any errors for added file.
  S.runWithAST("", Added, [&](llvm::Expected<InputsAndAST> AST) {
    EXPECT_TRUE(bool(AST));
  });
  S.runWithPreamble("", Added, [&](llvm::Expected<InputsAndPreamble> Preamble) {
    EXPECT_TRUE(bool(Preamble));
  });
  S.remove(Added);

  // Assert that all operations fail after removing the file.
  S.runWithAST("", Added, [&](llvm::Expected<InputsAndAST> AST) {
    ASSERT_FALSE(bool(AST));
    ignoreError(AST.takeError());
  });
  S.runWithPreamble("", Added, [&](llvm::Expected<InputsAndPreamble> Preamble) {
    ASSERT_FALSE(bool(Preamble));
    ignoreError(Preamble.takeError());
  });
  // remove() shouldn't crash on missing files.
  S.remove(Added);
}

TEST_F(TUSchedulerTests, WantDiagnostics) {
  std::atomic<int> CallbackCount(0);
  {
    // To avoid a racy test, don't allow tasks to actualy run on the worker
    // thread until we've scheduled them all.
    Notification Ready;
    TUScheduler S(
        getDefaultAsyncThreadsCount(),
        /*StorePreamblesInMemory=*/true,
        /*PreambleParsedCallback=*/nullptr,
        /*UpdateDebounce=*/std::chrono::steady_clock::duration::zero(),
        ASTRetentionPolicy());
    auto Path = testPath("foo.cpp");
    S.update(Path, getInputs(Path, ""), WantDiagnostics::Yes,
             [&](std::vector<Diag>) { Ready.wait(); });

    S.update(Path, getInputs(Path, "request diags"), WantDiagnostics::Yes,
             [&](std::vector<Diag> Diags) { ++CallbackCount; });
    S.update(Path, getInputs(Path, "auto (clobbered)"), WantDiagnostics::Auto,
             [&](std::vector<Diag> Diags) {
               ADD_FAILURE() << "auto should have been cancelled by auto";
             });
    S.update(Path, getInputs(Path, "request no diags"), WantDiagnostics::No,
             [&](std::vector<Diag> Diags) {
               ADD_FAILURE() << "no diags should not be called back";
             });
    S.update(Path, getInputs(Path, "auto (produces)"), WantDiagnostics::Auto,
             [&](std::vector<Diag> Diags) { ++CallbackCount; });
    Ready.notify();
  }
  EXPECT_EQ(2, CallbackCount);
}

TEST_F(TUSchedulerTests, Debounce) {
  std::atomic<int> CallbackCount(0);
  {
    TUScheduler S(getDefaultAsyncThreadsCount(),
                  /*StorePreamblesInMemory=*/true,
                  /*PreambleParsedCallback=*/nullptr,
                  /*UpdateDebounce=*/std::chrono::seconds(1),
                  ASTRetentionPolicy());
    // FIXME: we could probably use timeouts lower than 1 second here.
    auto Path = testPath("foo.cpp");
    S.update(Path, getInputs(Path, "auto (debounced)"), WantDiagnostics::Auto,
             [&](std::vector<Diag> Diags) {
               ADD_FAILURE() << "auto should have been debounced and canceled";
             });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    S.update(Path, getInputs(Path, "auto (timed out)"), WantDiagnostics::Auto,
             [&](std::vector<Diag> Diags) { ++CallbackCount; });
    std::this_thread::sleep_for(std::chrono::seconds(2));
    S.update(Path, getInputs(Path, "auto (shut down)"), WantDiagnostics::Auto,
             [&](std::vector<Diag> Diags) { ++CallbackCount; });
  }
  EXPECT_EQ(2, CallbackCount);
}

TEST_F(TUSchedulerTests, ManyUpdates) {
  const int FilesCount = 3;
  const int UpdatesPerFile = 10;

  std::mutex Mut;
  int TotalASTReads = 0;
  int TotalPreambleReads = 0;
  int TotalUpdates = 0;

  // Run TUScheduler and collect some stats.
  {
    TUScheduler S(getDefaultAsyncThreadsCount(),
                  /*StorePreamblesInMemory=*/true,
                  /*PreambleParsedCallback=*/nullptr,
                  /*UpdateDebounce=*/std::chrono::milliseconds(50),
                  ASTRetentionPolicy());

    std::vector<std::string> Files;
    for (int I = 0; I < FilesCount; ++I) {
      std::string Name = "foo" + std::to_string(I) + ".cpp";
      Files.push_back(testPath(Name));
      this->Files[Files.back()] = "";
    }

    llvm::StringRef Contents1 = R"cpp(int a;)cpp";
    llvm::StringRef Contents2 = R"cpp(int main() { return 1; })cpp";
    llvm::StringRef Contents3 =
        R"cpp(int a; int b; int sum() { return a + b; })cpp";

    llvm::StringRef AllContents[] = {Contents1, Contents2, Contents3};
    const int AllContentsSize = 3;

    // Scheduler may run tasks asynchronously, but should propagate the context.
    // We stash a nonce in the context, and verify it in the task.
    static Key<int> NonceKey;
    int Nonce = 0;

    for (int FileI = 0; FileI < FilesCount; ++FileI) {
      for (int UpdateI = 0; UpdateI < UpdatesPerFile; ++UpdateI) {
        auto Contents = AllContents[(FileI + UpdateI) % AllContentsSize];

        auto File = Files[FileI];
        auto Inputs = getInputs(File, Contents.str());

        {
          WithContextValue WithNonce(NonceKey, ++Nonce);
          S.update(File, Inputs, WantDiagnostics::Auto,
                   [Nonce, &Mut,
                    &TotalUpdates](llvm::Optional<std::vector<Diag>> Diags) {
                     EXPECT_THAT(Context::current().get(NonceKey),
                                 Pointee(Nonce));

                     std::lock_guard<std::mutex> Lock(Mut);
                     ++TotalUpdates;
                   });
        }

        {
          WithContextValue WithNonce(NonceKey, ++Nonce);
          S.runWithAST("CheckAST", File,
                       [Inputs, Nonce, &Mut,
                        &TotalASTReads](llvm::Expected<InputsAndAST> AST) {
                         EXPECT_THAT(Context::current().get(NonceKey),
                                     Pointee(Nonce));

                         ASSERT_TRUE((bool)AST);
                         EXPECT_EQ(AST->Inputs.FS, Inputs.FS);
                         EXPECT_EQ(AST->Inputs.Contents, Inputs.Contents);

                         std::lock_guard<std::mutex> Lock(Mut);
                         ++TotalASTReads;
                       });
        }

        {
          WithContextValue WithNonce(NonceKey, ++Nonce);
          S.runWithPreamble("CheckPreamble", File,
                            [Inputs, Nonce, &Mut, &TotalPreambleReads](
                                llvm::Expected<InputsAndPreamble> Preamble) {
                              EXPECT_THAT(Context::current().get(NonceKey),
                                          Pointee(Nonce));

                              ASSERT_TRUE((bool)Preamble);
                              EXPECT_EQ(Preamble->Contents, Inputs.Contents);

                              std::lock_guard<std::mutex> Lock(Mut);
                              ++TotalPreambleReads;
                            });
        }
      }
    }
  } // TUScheduler destructor waits for all operations to finish.

  std::lock_guard<std::mutex> Lock(Mut);
  EXPECT_EQ(TotalUpdates, FilesCount * UpdatesPerFile);
  EXPECT_EQ(TotalASTReads, FilesCount * UpdatesPerFile);
  EXPECT_EQ(TotalPreambleReads, FilesCount * UpdatesPerFile);
}

TEST_F(TUSchedulerTests, EvictedAST) {
  std::atomic<int> BuiltASTCounter(0);
  ASTRetentionPolicy Policy;
  Policy.MaxRetainedASTs = 2;
  TUScheduler S(
      /*AsyncThreadsCount=*/1, /*StorePreambleInMemory=*/true,
      PreambleParsedCallback(),
      /*UpdateDebounce=*/std::chrono::steady_clock::duration::zero(), Policy);

  llvm::StringLiteral SourceContents = R"cpp(
    int* a;
    double* b = a;
  )cpp";
  llvm::StringLiteral OtherSourceContents = R"cpp(
    int* a;
    double* b = a + 0;
  )cpp";

  auto Foo = testPath("foo.cpp");
  auto Bar = testPath("bar.cpp");
  auto Baz = testPath("baz.cpp");

  // Build one file in advance. We will not access it later, so it will be the
  // one that the cache will evict.
  S.update(Foo, getInputs(Foo, SourceContents), WantDiagnostics::Yes,
           [&BuiltASTCounter](std::vector<Diag> Diags) { ++BuiltASTCounter; });
  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(1)));
  ASSERT_EQ(BuiltASTCounter.load(), 1);

  // Build two more files. Since we can retain only 2 ASTs, these should be the
  // ones we see in the cache later.
  S.update(Bar, getInputs(Bar, SourceContents), WantDiagnostics::Yes,
           [&BuiltASTCounter](std::vector<Diag> Diags) { ++BuiltASTCounter; });
  S.update(Baz, getInputs(Baz, SourceContents), WantDiagnostics::Yes,
           [&BuiltASTCounter](std::vector<Diag> Diags) { ++BuiltASTCounter; });
  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(1)));
  ASSERT_EQ(BuiltASTCounter.load(), 3);

  // Check only the last two ASTs are retained.
  ASSERT_THAT(S.getFilesWithCachedAST(), UnorderedElementsAre(Bar, Baz));

  // Access the old file again.
  S.update(Foo, getInputs(Foo, OtherSourceContents), WantDiagnostics::Yes,
           [&BuiltASTCounter](std::vector<Diag> Diags) { ++BuiltASTCounter; });
  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(1)));
  ASSERT_EQ(BuiltASTCounter.load(), 4);

  // Check the AST for foo.cpp is retained now and one of the others got
  // evicted.
  EXPECT_THAT(S.getFilesWithCachedAST(),
              UnorderedElementsAre(Foo, AnyOf(Bar, Baz)));
}

TEST_F(TUSchedulerTests, RunWaitsForPreamble) {
  // Testing strategy: we update the file and schedule a few preamble reads at
  // the same time. All reads should get the same non-null preamble.
  TUScheduler S(
      /*AsyncThreadsCount=*/4, /*StorePreambleInMemory=*/true,
      PreambleParsedCallback(),
      /*UpdateDebounce=*/std::chrono::steady_clock::duration::zero(),
      ASTRetentionPolicy());
  auto Foo = testPath("foo.cpp");
  auto NonEmptyPreamble = R"cpp(
    #define FOO 1
    #define BAR 2

    int main() {}
  )cpp";
  constexpr int ReadsToSchedule = 10;
  std::mutex PreamblesMut;
  std::vector<const void *> Preambles(ReadsToSchedule, nullptr);
  S.update(Foo, getInputs(Foo, NonEmptyPreamble), WantDiagnostics::Auto,
           [](std::vector<Diag>) {});
  for (int I = 0; I < ReadsToSchedule; ++I) {
    S.runWithPreamble(
        "test", Foo,
        [I, &PreamblesMut, &Preambles](llvm::Expected<InputsAndPreamble> IP) {
          std::lock_guard<std::mutex> Lock(PreamblesMut);
          Preambles[I] = cantFail(std::move(IP)).Preamble;
        });
  }
  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
  // Check all actions got the same non-null preamble.
  std::lock_guard<std::mutex> Lock(PreamblesMut);
  ASSERT_NE(Preambles[0], nullptr);
  ASSERT_THAT(Preambles, Each(Preambles[0]));
}

TEST_F(TUSchedulerTests, NoopOnEmptyChanges) {
  TUScheduler S(
      /*AsyncThreadsCount=*/getDefaultAsyncThreadsCount(),
      /*StorePreambleInMemory=*/true, PreambleParsedCallback(),
      /*UpdateDebounce=*/std::chrono::steady_clock::duration::zero(),
      ASTRetentionPolicy());

  auto Source = testPath("foo.cpp");
  auto Header = testPath("foo.h");

  Files[Header] = "int a;";
  Timestamps[Header] = time_t(0);

  auto SourceContents = R"cpp(
      #include "foo.h"
      int b = a;
    )cpp";

  // Return value indicates if the updated callback was received.
  auto DoUpdate = [&](ParseInputs Inputs) -> bool {
    std::atomic<bool> Updated(false);
    Updated = false;
    S.update(Source, std::move(Inputs), WantDiagnostics::Yes,
             [&Updated](std::vector<Diag>) { Updated = true; });
    bool UpdateFinished = S.blockUntilIdle(timeoutSeconds(1));
    if (!UpdateFinished)
      ADD_FAILURE() << "Updated has not finished in one second. Threading bug?";
    return Updated;
  };

  // Test that subsequent updates with the same inputs do not cause rebuilds.
  ASSERT_TRUE(DoUpdate(getInputs(Source, SourceContents)));
  ASSERT_FALSE(DoUpdate(getInputs(Source, SourceContents)));

  // Update to a header should cause a rebuild, though.
  Files[Header] = time_t(1);
  ASSERT_TRUE(DoUpdate(getInputs(Source, SourceContents)));
  ASSERT_FALSE(DoUpdate(getInputs(Source, SourceContents)));

  // Update to the contents should cause a rebuild.
  auto OtherSourceContents = R"cpp(
      #include "foo.h"
      int c = d;
    )cpp";
  ASSERT_TRUE(DoUpdate(getInputs(Source, OtherSourceContents)));
  ASSERT_FALSE(DoUpdate(getInputs(Source, OtherSourceContents)));

  // Update to the compile commands should also cause a rebuild.
  CDB.ExtraClangFlags.push_back("-DSOMETHING");
  ASSERT_TRUE(DoUpdate(getInputs(Source, OtherSourceContents)));
  ASSERT_FALSE(DoUpdate(getInputs(Source, OtherSourceContents)));
}

TEST_F(TUSchedulerTests, NoChangeDiags) {
  TUScheduler S(
      /*AsyncThreadsCount=*/getDefaultAsyncThreadsCount(),
      /*StorePreambleInMemory=*/true, PreambleParsedCallback(),
      /*UpdateDebounce=*/std::chrono::steady_clock::duration::zero(),
      ASTRetentionPolicy());

  auto FooCpp = testPath("foo.cpp");
  auto Contents = "int a; int b;";

  S.update(FooCpp, getInputs(FooCpp, Contents), WantDiagnostics::No,
           [](std::vector<Diag>) { ADD_FAILURE() << "Should not be called."; });
  S.runWithAST("touchAST", FooCpp, [](llvm::Expected<InputsAndAST> IA) {
    // Make sure the AST was actually built.
    cantFail(std::move(IA));
  });
  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(1)));

  // Even though the inputs didn't change and AST can be reused, we need to
  // report the diagnostics, as they were not reported previously.
  std::atomic<bool> SeenDiags(false);
  S.update(FooCpp, getInputs(FooCpp, Contents), WantDiagnostics::Auto,
           [&](std::vector<Diag>) { SeenDiags = true; });
  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(1)));
  ASSERT_TRUE(SeenDiags);

  // Subsequent request does not get any diagnostics callback because the same
  // diags have previously been reported and the inputs didn't change.
  S.update(
      FooCpp, getInputs(FooCpp, Contents), WantDiagnostics::Auto,
      [&](std::vector<Diag>) { ADD_FAILURE() << "Should not be called."; });
  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(1)));
}

} // namespace clangd
} // namespace clang
