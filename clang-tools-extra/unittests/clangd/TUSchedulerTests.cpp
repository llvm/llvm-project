//===-- TUSchedulerTests.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "Context.h"
#include "Matchers.h"
#include "TUScheduler.h"
#include "TestFS.h"
#include "gmock/gmock.h"
#include "llvm/ADT/ScopeExit.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <utility>

using namespace llvm;
namespace clang {
namespace clangd {
namespace {

using ::testing::_;
using ::testing::AllOf;
using ::testing::AnyOf;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::Pair;
using ::testing::Pointee;
using ::testing::UnorderedElementsAre;

MATCHER_P2(TUState, State, ActionName, "") {
  return arg.Action.S == State && arg.Action.Name == ActionName;
}

class TUSchedulerTests : public ::testing::Test {
protected:
  ParseInputs getInputs(PathRef File, std::string Contents) {
    return ParseInputs{*CDB.getCompileCommand(File),
                       buildTestFS(Files, Timestamps), std::move(Contents)};
  }

  void updateWithCallback(TUScheduler &S, PathRef File, StringRef Contents,
                          WantDiagnostics WD,
                          llvm::unique_function<void()> CB) {
    WithContextValue Ctx(llvm::make_scope_exit(std::move(CB)));
    S.update(File, getInputs(File, Contents), WD);
  }

  static Key<llvm::unique_function<void(PathRef File, std::vector<Diag>)>>
      DiagsCallbackKey;

  /// A diagnostics callback that should be passed to TUScheduler when it's used
  /// in updateWithDiags.
  static std::unique_ptr<ParsingCallbacks> captureDiags() {
    class CaptureDiags : public ParsingCallbacks {
      void onDiagnostics(PathRef File, std::vector<Diag> Diags) override {
        auto D = Context::current().get(DiagsCallbackKey);
        if (!D)
          return;
        const_cast<llvm::unique_function<void(PathRef, std::vector<Diag>)> &> (
            *D)(File, Diags);
      }
    };
    return llvm::make_unique<CaptureDiags>();
  }

  /// Schedule an update and call \p CB with the diagnostics it produces, if
  /// any. The TUScheduler should be created with captureDiags as a
  /// DiagsCallback for this to work.
  void updateWithDiags(TUScheduler &S, PathRef File, ParseInputs Inputs,
                       WantDiagnostics WD,
                       llvm::unique_function<void(std::vector<Diag>)> CB) {
    Path OrigFile = File.str();
    WithContextValue Ctx(
        DiagsCallbackKey,
        Bind(
            [OrigFile](decltype(CB) CB, PathRef File, std::vector<Diag> Diags) {
              assert(File == OrigFile);
              CB(std::move(Diags));
            },
            std::move(CB)));
    S.update(File, std::move(Inputs), WD);
  }

  void updateWithDiags(TUScheduler &S, PathRef File, llvm::StringRef Contents,
                       WantDiagnostics WD,
                       llvm::unique_function<void(std::vector<Diag>)> CB) {
    return updateWithDiags(S, File, getInputs(File, Contents), WD,
                           std::move(CB));
  }

  StringMap<std::string> Files;
  StringMap<time_t> Timestamps;
  MockCompilationDatabase CDB;
};

Key<llvm::unique_function<void(PathRef File, std::vector<Diag>)>>
    TUSchedulerTests::DiagsCallbackKey;

TEST_F(TUSchedulerTests, MissingFiles) {
  TUScheduler S(getDefaultAsyncThreadsCount(),
                /*StorePreamblesInMemory=*/true, /*ASTCallbacks=*/nullptr,
                /*UpdateDebounce=*/std::chrono::steady_clock::duration::zero(),
                ASTRetentionPolicy());

  auto Added = testPath("added.cpp");
  Files[Added] = "";

  auto Missing = testPath("missing.cpp");
  Files[Missing] = "";

  S.update(Added, getInputs(Added, ""), WantDiagnostics::No);

  // Assert each operation for missing file is an error (even if it's available
  // in VFS).
  S.runWithAST("", Missing,
               [&](Expected<InputsAndAST> AST) { EXPECT_ERROR(AST); });
  S.runWithPreamble(
      "", Missing, TUScheduler::Stale,
      [&](Expected<InputsAndPreamble> Preamble) { EXPECT_ERROR(Preamble); });
  // remove() shouldn't crash on missing files.
  S.remove(Missing);

  // Assert there aren't any errors for added file.
  S.runWithAST("", Added,
               [&](Expected<InputsAndAST> AST) { EXPECT_TRUE(bool(AST)); });
  S.runWithPreamble("", Added, TUScheduler::Stale,
                    [&](Expected<InputsAndPreamble> Preamble) {
                      EXPECT_TRUE(bool(Preamble));
                    });
  S.remove(Added);

  // Assert that all operations fail after removing the file.
  S.runWithAST("", Added,
               [&](Expected<InputsAndAST> AST) { EXPECT_ERROR(AST); });
  S.runWithPreamble("", Added, TUScheduler::Stale,
                    [&](Expected<InputsAndPreamble> Preamble) {
                      ASSERT_FALSE(bool(Preamble));
                      llvm::consumeError(Preamble.takeError());
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
        /*StorePreamblesInMemory=*/true, captureDiags(),
        /*UpdateDebounce=*/std::chrono::steady_clock::duration::zero(),
        ASTRetentionPolicy());
    auto Path = testPath("foo.cpp");
    updateWithDiags(S, Path, "", WantDiagnostics::Yes,
                    [&](std::vector<Diag>) { Ready.wait(); });
    updateWithDiags(S, Path, "request diags", WantDiagnostics::Yes,
                    [&](std::vector<Diag>) { ++CallbackCount; });
    updateWithDiags(S, Path, "auto (clobbered)", WantDiagnostics::Auto,
                    [&](std::vector<Diag>) {
                      ADD_FAILURE()
                          << "auto should have been cancelled by auto";
                    });
    updateWithDiags(S, Path, "request no diags", WantDiagnostics::No,
                    [&](std::vector<Diag>) {
                      ADD_FAILURE() << "no diags should not be called back";
                    });
    updateWithDiags(S, Path, "auto (produces)", WantDiagnostics::Auto,
                    [&](std::vector<Diag>) { ++CallbackCount; });
    Ready.notify();

    ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
  }
  EXPECT_EQ(2, CallbackCount);
}

TEST_F(TUSchedulerTests, Debounce) {
  std::atomic<int> CallbackCount(0);
  {
    TUScheduler S(getDefaultAsyncThreadsCount(),
                  /*StorePreamblesInMemory=*/true, captureDiags(),
                  /*UpdateDebounce=*/std::chrono::seconds(1),
                  ASTRetentionPolicy());
    // FIXME: we could probably use timeouts lower than 1 second here.
    auto Path = testPath("foo.cpp");
    updateWithDiags(S, Path, "auto (debounced)", WantDiagnostics::Auto,
                    [&](std::vector<Diag>) {
                      ADD_FAILURE()
                          << "auto should have been debounced and canceled";
                    });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    updateWithDiags(S, Path, "auto (timed out)", WantDiagnostics::Auto,
                    [&](std::vector<Diag>) { ++CallbackCount; });
    std::this_thread::sleep_for(std::chrono::seconds(2));
    updateWithDiags(S, Path, "auto (shut down)", WantDiagnostics::Auto,
                    [&](std::vector<Diag>) { ++CallbackCount; });

    ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
  }
  EXPECT_EQ(2, CallbackCount);
}

static std::vector<std::string> includes(const PreambleData *Preamble) {
  std::vector<std::string> Result;
  if (Preamble)
    for (const auto &Inclusion : Preamble->Includes.MainFileIncludes)
      Result.push_back(Inclusion.Written);
  return Result;
}

TEST_F(TUSchedulerTests, PreambleConsistency) {
  std::atomic<int> CallbackCount(0);
  {
    Notification InconsistentReadDone; // Must live longest.
    TUScheduler S(
        getDefaultAsyncThreadsCount(), /*StorePreamblesInMemory=*/true,
        /*ASTCallbacks=*/nullptr,
        /*UpdateDebounce=*/std::chrono::steady_clock::duration::zero(),
        ASTRetentionPolicy());
    auto Path = testPath("foo.cpp");
    // Schedule two updates (A, B) and two preamble reads (stale, consistent).
    // The stale read should see A, and the consistent read should see B.
    // (We recognize the preambles by their included files).
    updateWithCallback(S, Path, "#include <A>", WantDiagnostics::Yes, [&]() {
      // This callback runs in between the two preamble updates.

      // This blocks update B, preventing it from winning the race
      // against the stale read.
      // If the first read was instead consistent, this would deadlock.
      InconsistentReadDone.wait();
      // This delays update B, preventing it from winning a race
      // against the consistent read. The consistent read sees B
      // only because it waits for it.
      // If the second read was stale, it would usually see A.
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    });
    S.update(Path, getInputs(Path, "#include <B>"), WantDiagnostics::Yes);

    S.runWithPreamble("StaleRead", Path, TUScheduler::Stale,
                      [&](Expected<InputsAndPreamble> Pre) {
                        ASSERT_TRUE(bool(Pre));
                        assert(bool(Pre));
                        EXPECT_THAT(includes(Pre->Preamble),
                                    ElementsAre("<A>"));
                        InconsistentReadDone.notify();
                        ++CallbackCount;
                      });
    S.runWithPreamble("ConsistentRead", Path, TUScheduler::Consistent,
                      [&](Expected<InputsAndPreamble> Pre) {
                        ASSERT_TRUE(bool(Pre));
                        EXPECT_THAT(includes(Pre->Preamble),
                                    ElementsAre("<B>"));
                        ++CallbackCount;
                      });
  }
  EXPECT_EQ(2, CallbackCount);
}

TEST_F(TUSchedulerTests, Cancellation) {
  // We have the following update/read sequence
  //   U0
  //   U1(WantDiags=Yes) <-- cancelled
  //    R1               <-- cancelled
  //   U2(WantDiags=Yes) <-- cancelled
  //    R2A              <-- cancelled
  //    R2B
  //   U3(WantDiags=Yes)
  //    R3               <-- cancelled
  std::vector<std::string> DiagsSeen, ReadsSeen, ReadsCanceled;
  {
    Notification Proceed; // Ensure we schedule everything.
    TUScheduler S(
        getDefaultAsyncThreadsCount(), /*StorePreamblesInMemory=*/true,
        /*ASTCallbacks=*/captureDiags(),
        /*UpdateDebounce=*/std::chrono::steady_clock::duration::zero(),
        ASTRetentionPolicy());
    auto Path = testPath("foo.cpp");
    // Helper to schedule a named update and return a function to cancel it.
    auto Update = [&](std::string ID) -> Canceler {
      auto T = cancelableTask();
      WithContext C(std::move(T.first));
      updateWithDiags(
          S, Path, "//" + ID, WantDiagnostics::Yes,
          [&, ID](std::vector<Diag> Diags) { DiagsSeen.push_back(ID); });
      return std::move(T.second);
    };
    // Helper to schedule a named read and return a function to cancel it.
    auto Read = [&](std::string ID) -> Canceler {
      auto T = cancelableTask();
      WithContext C(std::move(T.first));
      S.runWithAST(ID, Path, [&, ID](llvm::Expected<InputsAndAST> E) {
        if (auto Err = E.takeError()) {
          if (Err.isA<CancelledError>()) {
            ReadsCanceled.push_back(ID);
            consumeError(std::move(Err));
          } else {
            ADD_FAILURE() << "Non-cancelled error for " << ID << ": "
                          << llvm::toString(std::move(Err));
          }
        } else {
          ReadsSeen.push_back(ID);
        }
      });
      return std::move(T.second);
    };

    updateWithCallback(S, Path, "", WantDiagnostics::Yes,
                       [&]() { Proceed.wait(); });
    // The second parens indicate cancellation, where present.
    Update("U1")();
    Read("R1")();
    Update("U2")();
    Read("R2A")();
    Read("R2B");
    Update("U3");
    Read("R3")();
    Proceed.notify();

    ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
  }
  EXPECT_THAT(DiagsSeen, ElementsAre("U2", "U3"))
      << "U1 and all dependent reads were cancelled. "
         "U2 has a dependent read R2A. "
         "U3 was not cancelled.";
  EXPECT_THAT(ReadsSeen, ElementsAre("R2B"))
      << "All reads other than R2B were cancelled";
  EXPECT_THAT(ReadsCanceled, ElementsAre("R1", "R2A", "R3"))
      << "All reads other than R2B were cancelled";
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
                  /*StorePreamblesInMemory=*/true, captureDiags(),
                  /*UpdateDebounce=*/std::chrono::milliseconds(50),
                  ASTRetentionPolicy());

    std::vector<std::string> Files;
    for (int I = 0; I < FilesCount; ++I) {
      std::string Name = "foo" + std::to_string(I) + ".cpp";
      Files.push_back(testPath(Name));
      this->Files[Files.back()] = "";
    }

    StringRef Contents1 = R"cpp(int a;)cpp";
    StringRef Contents2 = R"cpp(int main() { return 1; })cpp";
    StringRef Contents3 = R"cpp(int a; int b; int sum() { return a + b; })cpp";

    StringRef AllContents[] = {Contents1, Contents2, Contents3};
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
          updateWithDiags(
              S, File, Inputs, WantDiagnostics::Auto,
              [File, Nonce, &Mut, &TotalUpdates](std::vector<Diag>) {
                EXPECT_THAT(Context::current().get(NonceKey), Pointee(Nonce));

                std::lock_guard<std::mutex> Lock(Mut);
                ++TotalUpdates;
                EXPECT_EQ(File, *TUScheduler::getFileBeingProcessedInContext());
              });
        }
        {
          WithContextValue WithNonce(NonceKey, ++Nonce);
          S.runWithAST("CheckAST", File,
                       [File, Inputs, Nonce, &Mut,
                        &TotalASTReads](Expected<InputsAndAST> AST) {
                         EXPECT_THAT(Context::current().get(NonceKey),
                                     Pointee(Nonce));

                         ASSERT_TRUE((bool)AST);
                         EXPECT_EQ(AST->Inputs.FS, Inputs.FS);
                         EXPECT_EQ(AST->Inputs.Contents, Inputs.Contents);

                         std::lock_guard<std::mutex> Lock(Mut);
                         ++TotalASTReads;
                         EXPECT_EQ(
                             File,
                             *TUScheduler::getFileBeingProcessedInContext());
                       });
        }

        {
          WithContextValue WithNonce(NonceKey, ++Nonce);
          S.runWithPreamble(
              "CheckPreamble", File, TUScheduler::Stale,
              [File, Inputs, Nonce, &Mut,
               &TotalPreambleReads](Expected<InputsAndPreamble> Preamble) {
                EXPECT_THAT(Context::current().get(NonceKey), Pointee(Nonce));

                ASSERT_TRUE((bool)Preamble);
                EXPECT_EQ(Preamble->Contents, Inputs.Contents);

                std::lock_guard<std::mutex> Lock(Mut);
                ++TotalPreambleReads;
                EXPECT_EQ(File, *TUScheduler::getFileBeingProcessedInContext());
              });
        }
      }
    }
    ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
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
      /*ASTCallbacks=*/nullptr,
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
  updateWithCallback(S, Foo, SourceContents, WantDiagnostics::Yes,
                     [&BuiltASTCounter]() { ++BuiltASTCounter; });
  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
  ASSERT_EQ(BuiltASTCounter.load(), 1);

  // Build two more files. Since we can retain only 2 ASTs, these should be the
  // ones we see in the cache later.
  updateWithCallback(S, Bar, SourceContents, WantDiagnostics::Yes,
                     [&BuiltASTCounter]() { ++BuiltASTCounter; });
  updateWithCallback(S, Baz, SourceContents, WantDiagnostics::Yes,
                     [&BuiltASTCounter]() { ++BuiltASTCounter; });
  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
  ASSERT_EQ(BuiltASTCounter.load(), 3);

  // Check only the last two ASTs are retained.
  ASSERT_THAT(S.getFilesWithCachedAST(), UnorderedElementsAre(Bar, Baz));

  // Access the old file again.
  updateWithCallback(S, Foo, OtherSourceContents, WantDiagnostics::Yes,
                     [&BuiltASTCounter]() { ++BuiltASTCounter; });
  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
  ASSERT_EQ(BuiltASTCounter.load(), 4);

  // Check the AST for foo.cpp is retained now and one of the others got
  // evicted.
  EXPECT_THAT(S.getFilesWithCachedAST(),
              UnorderedElementsAre(Foo, AnyOf(Bar, Baz)));
}

TEST_F(TUSchedulerTests, EmptyPreamble) {
  TUScheduler S(
      /*AsyncThreadsCount=*/4, /*StorePreambleInMemory=*/true,
      /*ASTCallbacks=*/nullptr,
      /*UpdateDebounce=*/std::chrono::steady_clock::duration::zero(),
      ASTRetentionPolicy());

  auto Foo = testPath("foo.cpp");
  auto Header = testPath("foo.h");

  Files[Header] = "void foo()";
  Timestamps[Header] = time_t(0);
  auto WithPreamble = R"cpp(
    #include "foo.h"
    int main() {}
  )cpp";
  auto WithEmptyPreamble = R"cpp(int main() {})cpp";
  S.update(Foo, getInputs(Foo, WithPreamble), WantDiagnostics::Auto);
  S.runWithPreamble("getNonEmptyPreamble", Foo, TUScheduler::Stale,
                    [&](Expected<InputsAndPreamble> Preamble) {
                      // We expect to get a non-empty preamble.
                      EXPECT_GT(cantFail(std::move(Preamble))
                                    .Preamble->Preamble.getBounds()
                                    .Size,
                                0u);
                    });
  // Wait for the preamble is being built.
  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));

  // Update the file which results in an empty preamble.
  S.update(Foo, getInputs(Foo, WithEmptyPreamble), WantDiagnostics::Auto);
  // Wait for the preamble is being built.
  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
  S.runWithPreamble("getEmptyPreamble", Foo, TUScheduler::Stale,
                    [&](Expected<InputsAndPreamble> Preamble) {
                      // We expect to get an empty preamble.
                      EXPECT_EQ(cantFail(std::move(Preamble))
                                    .Preamble->Preamble.getBounds()
                                    .Size,
                                0u);
                    });
}

TEST_F(TUSchedulerTests, RunWaitsForPreamble) {
  // Testing strategy: we update the file and schedule a few preamble reads at
  // the same time. All reads should get the same non-null preamble.
  TUScheduler S(
      /*AsyncThreadsCount=*/4, /*StorePreambleInMemory=*/true,
      /*ASTCallbacks=*/nullptr,
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
  S.update(Foo, getInputs(Foo, NonEmptyPreamble), WantDiagnostics::Auto);
  for (int I = 0; I < ReadsToSchedule; ++I) {
    S.runWithPreamble(
        "test", Foo, TUScheduler::Stale,
        [I, &PreamblesMut, &Preambles](Expected<InputsAndPreamble> IP) {
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
      /*StorePreambleInMemory=*/true, captureDiags(),
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
  auto DoUpdate = [&](std::string Contents) -> bool {
    std::atomic<bool> Updated(false);
    Updated = false;
    updateWithDiags(S, Source, Contents, WantDiagnostics::Yes,
                    [&Updated](std::vector<Diag>) { Updated = true; });
    bool UpdateFinished = S.blockUntilIdle(timeoutSeconds(10));
    if (!UpdateFinished)
      ADD_FAILURE() << "Updated has not finished in one second. Threading bug?";
    return Updated;
  };

  // Test that subsequent updates with the same inputs do not cause rebuilds.
  ASSERT_TRUE(DoUpdate(SourceContents));
  ASSERT_FALSE(DoUpdate(SourceContents));

  // Update to a header should cause a rebuild, though.
  Timestamps[Header] = time_t(1);
  ASSERT_TRUE(DoUpdate(SourceContents));
  ASSERT_FALSE(DoUpdate(SourceContents));

  // Update to the contents should cause a rebuild.
  auto OtherSourceContents = R"cpp(
      #include "foo.h"
      int c = d;
    )cpp";
  ASSERT_TRUE(DoUpdate(OtherSourceContents));
  ASSERT_FALSE(DoUpdate(OtherSourceContents));

  // Update to the compile commands should also cause a rebuild.
  CDB.ExtraClangFlags.push_back("-DSOMETHING");
  ASSERT_TRUE(DoUpdate(OtherSourceContents));
  ASSERT_FALSE(DoUpdate(OtherSourceContents));
}

TEST_F(TUSchedulerTests, NoChangeDiags) {
  TUScheduler S(
      /*AsyncThreadsCount=*/getDefaultAsyncThreadsCount(),
      /*StorePreambleInMemory=*/true, captureDiags(),
      /*UpdateDebounce=*/std::chrono::steady_clock::duration::zero(),
      ASTRetentionPolicy());

  auto FooCpp = testPath("foo.cpp");
  auto Contents = "int a; int b;";

  updateWithDiags(
      S, FooCpp, Contents, WantDiagnostics::No,
      [](std::vector<Diag>) { ADD_FAILURE() << "Should not be called."; });
  S.runWithAST("touchAST", FooCpp, [](Expected<InputsAndAST> IA) {
    // Make sure the AST was actually built.
    cantFail(std::move(IA));
  });
  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));

  // Even though the inputs didn't change and AST can be reused, we need to
  // report the diagnostics, as they were not reported previously.
  std::atomic<bool> SeenDiags(false);
  updateWithDiags(S, FooCpp, Contents, WantDiagnostics::Auto,
                  [&](std::vector<Diag>) { SeenDiags = true; });
  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
  ASSERT_TRUE(SeenDiags);

  // Subsequent request does not get any diagnostics callback because the same
  // diags have previously been reported and the inputs didn't change.
  updateWithDiags(
      S, FooCpp, Contents, WantDiagnostics::Auto,
      [&](std::vector<Diag>) { ADD_FAILURE() << "Should not be called."; });
  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
}

TEST_F(TUSchedulerTests, Run) {
  TUScheduler S(/*AsyncThreadsCount=*/getDefaultAsyncThreadsCount(),
                /*StorePreambleInMemory=*/true, /*ASTCallbacks=*/nullptr,
                /*UpdateDebounce=*/std::chrono::steady_clock::duration::zero(),
                ASTRetentionPolicy());
  std::atomic<int> Counter(0);
  S.run("add 1", [&] { ++Counter; });
  S.run("add 2", [&] { Counter += 2; });
  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
  EXPECT_EQ(Counter.load(), 3);
}

TEST_F(TUSchedulerTests, TUStatus) {
  class CaptureTUStatus : public DiagnosticsConsumer {
  public:
    void onDiagnosticsReady(PathRef File,
                            std::vector<Diag> Diagnostics) override {}

    void onFileUpdated(PathRef File, const TUStatus &Status) override {
      std::lock_guard<std::mutex> Lock(Mutex);
      AllStatus.push_back(Status);
    }

    std::vector<TUStatus> AllStatus;

  private:
    std::mutex Mutex;
  } CaptureTUStatus;
  MockFSProvider FS;
  MockCompilationDatabase CDB;
  ClangdServer Server(CDB, FS, CaptureTUStatus, ClangdServer::optsForTest());
  Annotations Code("int m^ain () {}");

  // We schedule the following tasks in the queue:
  //   [Update] [GoToDefinition]
  Server.addDocument(testPath("foo.cpp"), Code.code(), WantDiagnostics::Yes);
  Server.findDefinitions(testPath("foo.cpp"), Code.point(),
                         [](Expected<std::vector<Location>> Result) {
                           ASSERT_TRUE((bool)Result);
                         });

  ASSERT_TRUE(Server.blockUntilIdleForTest());

  EXPECT_THAT(CaptureTUStatus.AllStatus,
              ElementsAre(
                  // Statuses of "Update" action.
                  TUState(TUAction::Queued, "Update"),
                  TUState(TUAction::RunningAction, "Update"),
                  TUState(TUAction::BuildingPreamble, "Update"),
                  TUState(TUAction::BuildingFile, "Update"),

                  // Statuses of "Definitions" action
                  TUState(TUAction::Queued, "Definitions"),
                  TUState(TUAction::RunningAction, "Definitions"),
                  TUState(TUAction::Idle, /*No action*/ "")));
}

} // namespace
} // namespace clangd
} // namespace clang
