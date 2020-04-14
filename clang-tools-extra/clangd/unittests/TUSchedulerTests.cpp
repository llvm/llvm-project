//===-- TUSchedulerTests.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "Cancellation.h"
#include "ClangdServer.h"
#include "Context.h"
#include "Diagnostics.h"
#include "Matchers.h"
#include "ParsedAST.h"
#include "Path.h"
#include "Preamble.h"
#include "TUScheduler.h"
#include "TestFS.h"
#include "Threading.h"
#include "clang/Basic/DiagnosticDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <utility>

namespace clang {
namespace clangd {
namespace {

using ::testing::AnyOf;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Field;
using ::testing::IsEmpty;
using ::testing::Pointee;
using ::testing::UnorderedElementsAre;

MATCHER_P2(TUState, PreambleActivity, ASTActivity, "") {
  if (arg.PreambleActivity != PreambleActivity) {
    *result_listener << "preamblestate is "
                     << static_cast<uint8_t>(arg.PreambleActivity);
    return false;
  }
  if (arg.ASTActivity.K != ASTActivity) {
    *result_listener << "aststate is " << arg.ASTActivity.K;
    return false;
  }
  return true;
}

TUScheduler::Options optsForTest() {
  return TUScheduler::Options(ClangdServer::optsForTest());
}

class TUSchedulerTests : public ::testing::Test {
protected:
  ParseInputs getInputs(PathRef File, std::string Contents) {
    ParseInputs Inputs;
    Inputs.CompileCommand = *CDB.getCompileCommand(File);
    Inputs.FS = buildTestFS(Files, Timestamps);
    Inputs.Contents = std::move(Contents);
    Inputs.Opts = ParseOptions();
    return Inputs;
  }

  void updateWithCallback(TUScheduler &S, PathRef File,
                          llvm::StringRef Contents, WantDiagnostics WD,
                          llvm::unique_function<void()> CB) {
    updateWithCallback(S, File, getInputs(File, std::string(Contents)), WD,
                       std::move(CB));
  }

  void updateWithCallback(TUScheduler &S, PathRef File, ParseInputs Inputs,
                          WantDiagnostics WD,
                          llvm::unique_function<void()> CB) {
    WithContextValue Ctx(llvm::make_scope_exit(std::move(CB)));
    S.update(File, Inputs, WD);
  }

  static Key<llvm::unique_function<void(PathRef File, std::vector<Diag>)>>
      DiagsCallbackKey;

  /// A diagnostics callback that should be passed to TUScheduler when it's used
  /// in updateWithDiags.
  static std::unique_ptr<ParsingCallbacks> captureDiags() {
    class CaptureDiags : public ParsingCallbacks {
    public:
      void onMainAST(PathRef File, ParsedAST &AST, PublishFn Publish) override {
        reportDiagnostics(File, AST.getDiagnostics(), Publish);
      }

      void onFailedAST(PathRef File, llvm::StringRef Version,
                       std::vector<Diag> Diags, PublishFn Publish) override {
        reportDiagnostics(File, Diags, Publish);
      }

    private:
      void reportDiagnostics(PathRef File, llvm::ArrayRef<Diag> Diags,
                             PublishFn Publish) {
        auto D = Context::current().get(DiagsCallbackKey);
        if (!D)
          return;
        Publish([&]() {
          const_cast<
              llvm::unique_function<void(PathRef, std::vector<Diag>)> &> (*D)(
              File, std::move(Diags));
        });
      }
    };
    return std::make_unique<CaptureDiags>();
  }

  /// Schedule an update and call \p CB with the diagnostics it produces, if
  /// any. The TUScheduler should be created with captureDiags as a
  /// DiagsCallback for this to work.
  void updateWithDiags(TUScheduler &S, PathRef File, ParseInputs Inputs,
                       WantDiagnostics WD,
                       llvm::unique_function<void(std::vector<Diag>)> CB) {
    Path OrigFile = File.str();
    WithContextValue Ctx(DiagsCallbackKey,
                         [OrigFile, CB = std::move(CB)](
                             PathRef File, std::vector<Diag> Diags) mutable {
                           assert(File == OrigFile);
                           CB(std::move(Diags));
                         });
    S.update(File, std::move(Inputs), WD);
  }

  void updateWithDiags(TUScheduler &S, PathRef File, llvm::StringRef Contents,
                       WantDiagnostics WD,
                       llvm::unique_function<void(std::vector<Diag>)> CB) {
    return updateWithDiags(S, File, getInputs(File, std::string(Contents)), WD,
                           std::move(CB));
  }

  llvm::StringMap<std::string> Files;
  llvm::StringMap<time_t> Timestamps;
  MockCompilationDatabase CDB;
};

Key<llvm::unique_function<void(PathRef File, std::vector<Diag>)>>
    TUSchedulerTests::DiagsCallbackKey;

TEST_F(TUSchedulerTests, MissingFiles) {
  TUScheduler S(CDB, optsForTest());

  auto Added = testPath("added.cpp");
  Files[Added] = "x";

  auto Missing = testPath("missing.cpp");
  Files[Missing] = "";

  S.update(Added, getInputs(Added, "x"), WantDiagnostics::No);

  // Assert each operation for missing file is an error (even if it's
  // available in VFS).
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
    // To avoid a racy test, don't allow tasks to actually run on the worker
    // thread until we've scheduled them all.
    Notification Ready;
    TUScheduler S(CDB, optsForTest(), captureDiags());
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
    auto Opts = optsForTest();
    Opts.UpdateDebounce = DebouncePolicy::fixed(std::chrono::seconds(1));
    TUScheduler S(CDB, Opts, captureDiags());
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
    TUScheduler S(CDB, optsForTest());
    auto Path = testPath("foo.cpp");
    // Schedule two updates (A, B) and two preamble reads (stale, consistent).
    // The stale read should see A, and the consistent read should see B.
    // (We recognize the preambles by their included files).
    auto Inputs = getInputs(Path, "#include <A>");
    Inputs.Version = "A";
    updateWithCallback(S, Path, Inputs, WantDiagnostics::Yes, [&]() {
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
    Inputs.Contents = "#include <B>";
    Inputs.Version = "B";
    S.update(Path, Inputs, WantDiagnostics::Yes);

    S.runWithPreamble("StaleRead", Path, TUScheduler::Stale,
                      [&](Expected<InputsAndPreamble> Pre) {
                        ASSERT_TRUE(bool(Pre));
                        ASSERT_TRUE(Pre->Preamble);
                        EXPECT_EQ(Pre->Preamble->Version, "A");
                        EXPECT_THAT(includes(Pre->Preamble),
                                    ElementsAre("<A>"));
                        InconsistentReadDone.notify();
                        ++CallbackCount;
                      });
    S.runWithPreamble("ConsistentRead", Path, TUScheduler::Consistent,
                      [&](Expected<InputsAndPreamble> Pre) {
                        ASSERT_TRUE(bool(Pre));
                        ASSERT_TRUE(Pre->Preamble);
                        EXPECT_EQ(Pre->Preamble->Version, "B");
                        EXPECT_THAT(includes(Pre->Preamble),
                                    ElementsAre("<B>"));
                        ++CallbackCount;
                      });
    S.blockUntilIdle(timeoutSeconds(10));
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
    TUScheduler S(CDB, optsForTest(), captureDiags());
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

TEST_F(TUSchedulerTests, InvalidationNoCrash) {
  auto Path = testPath("foo.cpp");
  TUScheduler S(CDB, optsForTest(), captureDiags());

  Notification StartedRunning;
  Notification ScheduledChange;
  // We expect invalidation logic to not crash by trying to invalidate a running
  // request.
  S.update(Path, getInputs(Path, ""), WantDiagnostics::Auto);
  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
  S.runWithAST(
      "invalidatable-but-running", Path,
      [&](llvm::Expected<InputsAndAST> AST) {
        StartedRunning.notify();
        ScheduledChange.wait();
        ASSERT_TRUE(bool(AST));
      },
      TUScheduler::InvalidateOnUpdate);
  StartedRunning.wait();
  S.update(Path, getInputs(Path, ""), WantDiagnostics::Auto);
  ScheduledChange.notify();
  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
}

TEST_F(TUSchedulerTests, Invalidation) {
  auto Path = testPath("foo.cpp");
  TUScheduler S(CDB, optsForTest(), captureDiags());
  std::atomic<int> Builds(0), Actions(0);

  Notification Start;
  updateWithDiags(S, Path, "a", WantDiagnostics::Yes, [&](std::vector<Diag>) {
    ++Builds;
    Start.wait();
  });
  S.runWithAST(
      "invalidatable", Path,
      [&](llvm::Expected<InputsAndAST> AST) {
        ++Actions;
        EXPECT_FALSE(bool(AST));
        llvm::Error E = AST.takeError();
        EXPECT_TRUE(E.isA<CancelledError>());
        handleAllErrors(std::move(E), [&](const CancelledError &E) {
          EXPECT_EQ(E.Reason, static_cast<int>(ErrorCode::ContentModified));
        });
      },
      TUScheduler::InvalidateOnUpdate);
  S.runWithAST(
      "not-invalidatable", Path,
      [&](llvm::Expected<InputsAndAST> AST) {
        ++Actions;
        EXPECT_TRUE(bool(AST));
      },
      TUScheduler::NoInvalidation);
  updateWithDiags(S, Path, "b", WantDiagnostics::Auto, [&](std::vector<Diag>) {
    ++Builds;
    ADD_FAILURE() << "Shouldn't build, all dependents invalidated";
  });
  S.runWithAST(
      "invalidatable", Path,
      [&](llvm::Expected<InputsAndAST> AST) {
        ++Actions;
        EXPECT_FALSE(bool(AST));
        llvm::Error E = AST.takeError();
        EXPECT_TRUE(E.isA<CancelledError>());
        consumeError(std::move(E));
      },
      TUScheduler::InvalidateOnUpdate);
  updateWithDiags(S, Path, "c", WantDiagnostics::Auto,
                  [&](std::vector<Diag>) { ++Builds; });
  S.runWithAST(
      "invalidatable", Path,
      [&](llvm::Expected<InputsAndAST> AST) {
        ++Actions;
        EXPECT_TRUE(bool(AST)) << "Shouldn't be invalidated, no update follows";
      },
      TUScheduler::InvalidateOnUpdate);
  Start.notify();
  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));

  EXPECT_EQ(2, Builds.load()) << "Middle build should be skipped";
  EXPECT_EQ(4, Actions.load()) << "All actions should run (some with error)";
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
    auto Opts = optsForTest();
    Opts.UpdateDebounce = DebouncePolicy::fixed(std::chrono::milliseconds(50));
    TUScheduler S(CDB, Opts, captureDiags());

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

    // Scheduler may run tasks asynchronously, but should propagate the
    // context. We stash a nonce in the context, and verify it in the task.
    static Key<int> NonceKey;
    int Nonce = 0;

    for (int FileI = 0; FileI < FilesCount; ++FileI) {
      for (int UpdateI = 0; UpdateI < UpdatesPerFile; ++UpdateI) {
        auto Contents = AllContents[(FileI + UpdateI) % AllContentsSize];

        auto File = Files[FileI];
        auto Inputs = getInputs(File, Contents.str());
        {
          WithContextValue WithNonce(NonceKey, ++Nonce);
          Inputs.Version = std::to_string(Nonce);
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
          S.runWithAST(
              "CheckAST", File,
              [File, Inputs, Nonce, &Mut,
               &TotalASTReads](Expected<InputsAndAST> AST) {
                EXPECT_THAT(Context::current().get(NonceKey), Pointee(Nonce));

                ASSERT_TRUE((bool)AST);
                EXPECT_EQ(AST->Inputs.FS, Inputs.FS);
                EXPECT_EQ(AST->Inputs.Contents, Inputs.Contents);
                EXPECT_EQ(AST->Inputs.Version, Inputs.Version);
                EXPECT_EQ(AST->AST.version(), Inputs.Version);

                std::lock_guard<std::mutex> Lock(Mut);
                ++TotalASTReads;
                EXPECT_EQ(File, *TUScheduler::getFileBeingProcessedInContext());
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
  auto Opts = optsForTest();
  Opts.AsyncThreadsCount = 1;
  Opts.RetentionPolicy.MaxRetainedASTs = 2;
  TUScheduler S(CDB, Opts);

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

  // Build two more files. Since we can retain only 2 ASTs, these should be
  // the ones we see in the cache later.
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

// We send "empty" changes to TUScheduler when we think some external event
// *might* have invalidated current state (e.g. a header was edited).
// Verify that this doesn't evict our cache entries.
TEST_F(TUSchedulerTests, NoopChangesDontThrashCache) {
  auto Opts = optsForTest();
  Opts.RetentionPolicy.MaxRetainedASTs = 1;
  TUScheduler S(CDB, Opts);

  auto Foo = testPath("foo.cpp");
  auto FooInputs = getInputs(Foo, "int x=1;");
  auto Bar = testPath("bar.cpp");
  auto BarInputs = getInputs(Bar, "int x=2;");

  // After opening Foo then Bar, AST cache contains Bar.
  S.update(Foo, FooInputs, WantDiagnostics::Auto);
  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
  S.update(Bar, BarInputs, WantDiagnostics::Auto);
  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
  ASSERT_THAT(S.getFilesWithCachedAST(), ElementsAre(Bar));

  // Any number of no-op updates to Foo don't dislodge Bar from the cache.
  S.update(Foo, FooInputs, WantDiagnostics::Auto);
  S.update(Foo, FooInputs, WantDiagnostics::Auto);
  S.update(Foo, FooInputs, WantDiagnostics::Auto);
  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
  ASSERT_THAT(S.getFilesWithCachedAST(), ElementsAre(Bar));
  // In fact each file has been built only once.
  ASSERT_EQ(S.fileStats().lookup(Foo).ASTBuilds, 1u);
  ASSERT_EQ(S.fileStats().lookup(Bar).ASTBuilds, 1u);
}

TEST_F(TUSchedulerTests, EmptyPreamble) {
  TUScheduler S(CDB, optsForTest());

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
  S.runWithPreamble(
      "getNonEmptyPreamble", Foo, TUScheduler::Stale,
      [&](Expected<InputsAndPreamble> Preamble) {
        // We expect to get a non-empty preamble.
        EXPECT_GT(
            cantFail(std::move(Preamble)).Preamble->Preamble.getBounds().Size,
            0u);
      });
  // Wait for the preamble is being built.
  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));

  // Update the file which results in an empty preamble.
  S.update(Foo, getInputs(Foo, WithEmptyPreamble), WantDiagnostics::Auto);
  // Wait for the preamble is being built.
  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
  S.runWithPreamble(
      "getEmptyPreamble", Foo, TUScheduler::Stale,
      [&](Expected<InputsAndPreamble> Preamble) {
        // We expect to get an empty preamble.
        EXPECT_EQ(
            cantFail(std::move(Preamble)).Preamble->Preamble.getBounds().Size,
            0u);
      });
}

TEST_F(TUSchedulerTests, RunWaitsForPreamble) {
  // Testing strategy: we update the file and schedule a few preamble reads at
  // the same time. All reads should get the same non-null preamble.
  TUScheduler S(CDB, optsForTest());
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
  TUScheduler S(CDB, optsForTest(), captureDiags());

  auto Source = testPath("foo.cpp");
  auto Header = testPath("foo.h");

  Files[Header] = "int a;";
  Timestamps[Header] = time_t(0);

  std::string SourceContents = R"cpp(
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
  ASSERT_EQ(S.fileStats().lookup(Source).ASTBuilds, 1u);
  ASSERT_EQ(S.fileStats().lookup(Source).PreambleBuilds, 1u);
  ASSERT_FALSE(DoUpdate(SourceContents));
  ASSERT_EQ(S.fileStats().lookup(Source).ASTBuilds, 1u);
  ASSERT_EQ(S.fileStats().lookup(Source).PreambleBuilds, 1u);

  // Update to a header should cause a rebuild, though.
  Timestamps[Header] = time_t(1);
  ASSERT_TRUE(DoUpdate(SourceContents));
  ASSERT_FALSE(DoUpdate(SourceContents));
  ASSERT_EQ(S.fileStats().lookup(Source).ASTBuilds, 2u);
  ASSERT_EQ(S.fileStats().lookup(Source).PreambleBuilds, 2u);

  // Update to the contents should cause a rebuild.
  SourceContents += "\nint c = b;";
  ASSERT_TRUE(DoUpdate(SourceContents));
  ASSERT_FALSE(DoUpdate(SourceContents));
  ASSERT_EQ(S.fileStats().lookup(Source).ASTBuilds, 3u);
  ASSERT_EQ(S.fileStats().lookup(Source).PreambleBuilds, 2u);

  // Update to the compile commands should also cause a rebuild.
  CDB.ExtraClangFlags.push_back("-DSOMETHING");
  ASSERT_TRUE(DoUpdate(SourceContents));
  ASSERT_FALSE(DoUpdate(SourceContents));
  ASSERT_EQ(S.fileStats().lookup(Source).ASTBuilds, 4u);
  ASSERT_EQ(S.fileStats().lookup(Source).PreambleBuilds, 3u);
}

TEST_F(TUSchedulerTests, ForceRebuild) {
  TUScheduler S(CDB, optsForTest(), captureDiags());

  auto Source = testPath("foo.cpp");
  auto Header = testPath("foo.h");

  auto SourceContents = R"cpp(
      #include "foo.h"
      int b = a;
    )cpp";

  ParseInputs Inputs = getInputs(Source, SourceContents);
  std::atomic<size_t> DiagCount(0);

  // Update the source contents, which should trigger an initial build with
  // the header file missing.
  updateWithDiags(
      S, Source, Inputs, WantDiagnostics::Yes,
      [&DiagCount](std::vector<Diag> Diags) {
        ++DiagCount;
        EXPECT_THAT(Diags,
                    ElementsAre(Field(&Diag::Message, "'foo.h' file not found"),
                                Field(&Diag::Message,
                                      "use of undeclared identifier 'a'")));
      });

  // Add the header file. We need to recreate the inputs since we changed a
  // file from underneath the test FS.
  Files[Header] = "int a;";
  Timestamps[Header] = time_t(1);
  Inputs = getInputs(Source, SourceContents);

  // The addition of the missing header file shouldn't trigger a rebuild since
  // we don't track missing files.
  updateWithDiags(
      S, Source, Inputs, WantDiagnostics::Yes,
      [&DiagCount](std::vector<Diag> Diags) {
        ++DiagCount;
        ADD_FAILURE() << "Did not expect diagnostics for missing header update";
      });

  // Forcing the reload should should cause a rebuild which no longer has any
  // errors.
  Inputs.ForceRebuild = true;
  updateWithDiags(S, Source, Inputs, WantDiagnostics::Yes,
                  [&DiagCount](std::vector<Diag> Diags) {
                    ++DiagCount;
                    EXPECT_THAT(Diags, IsEmpty());
                  });

  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
  EXPECT_EQ(DiagCount, 2U);
}
TEST_F(TUSchedulerTests, NoChangeDiags) {
  TUScheduler S(CDB, optsForTest(), captureDiags());

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
  TUScheduler S(CDB, optsForTest());
  std::atomic<int> Counter(0);
  S.run("add 1", [&] { ++Counter; });
  S.run("add 2", [&] { Counter += 2; });
  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
  EXPECT_EQ(Counter.load(), 3);

  Notification TaskRun;
  Key<int> TestKey;
  WithContextValue CtxWithKey(TestKey, 10);
  S.run("props context", [&] {
    EXPECT_EQ(Context::current().getExisting(TestKey), 10);
    TaskRun.notify();
  });
  TaskRun.wait();
}

TEST_F(TUSchedulerTests, TUStatus) {
  class CaptureTUStatus : public ClangdServer::Callbacks {
  public:
    void onFileUpdated(PathRef File, const TUStatus &Status) override {
      auto ASTAction = Status.ASTActivity.K;
      auto PreambleAction = Status.PreambleActivity;
      std::lock_guard<std::mutex> Lock(Mutex);
      // Only push the action if it has changed. Since TUStatus can be published
      // from either Preamble or AST thread and when one changes the other stays
      // the same.
      // Note that this can result in missing some updates when something other
      // than action kind changes, e.g. when AST is built/reused the action kind
      // stays as Building.
      if (ASTActions.empty() || ASTActions.back() != ASTAction)
        ASTActions.push_back(ASTAction);
      if (PreambleActions.empty() || PreambleActions.back() != PreambleAction)
        PreambleActions.push_back(PreambleAction);
    }

    std::vector<PreambleAction> preambleStatuses() {
      std::lock_guard<std::mutex> Lock(Mutex);
      return PreambleActions;
    }

    std::vector<ASTAction::Kind> astStatuses() {
      std::lock_guard<std::mutex> Lock(Mutex);
      return ASTActions;
    }

  private:
    std::mutex Mutex;
    std::vector<ASTAction::Kind> ASTActions;
    std::vector<PreambleAction> PreambleActions;
  } CaptureTUStatus;
  MockFSProvider FS;
  MockCompilationDatabase CDB;
  ClangdServer Server(CDB, FS, ClangdServer::optsForTest(), &CaptureTUStatus);
  Annotations Code("int m^ain () {}");

  // We schedule the following tasks in the queue:
  //   [Update] [GoToDefinition]
  Server.addDocument(testPath("foo.cpp"), Code.code(), "1",
                     WantDiagnostics::Auto);
  ASSERT_TRUE(Server.blockUntilIdleForTest());
  Server.locateSymbolAt(testPath("foo.cpp"), Code.point(),
                        [](Expected<std::vector<LocatedSymbol>> Result) {
                          ASSERT_TRUE((bool)Result);
                        });
  ASSERT_TRUE(Server.blockUntilIdleForTest());

  EXPECT_THAT(CaptureTUStatus.preambleStatuses(),
              ElementsAre(
                  // PreambleThread starts idle, as the update is first handled
                  // by ASTWorker.
                  PreambleAction::Idle,
                  // Then it starts building first preamble and releases that to
                  // ASTWorker.
                  PreambleAction::Building,
                  // Then goes idle and stays that way as we don't receive any
                  // more update requests.
                  PreambleAction::Idle));
  EXPECT_THAT(CaptureTUStatus.astStatuses(),
              ElementsAre(
                  // Starts handling the update action and blocks until the
                  // first preamble is built.
                  ASTAction::RunningAction,
                  // Afterwqards it builds an AST for that preamble to publish
                  // diagnostics.
                  ASTAction::Building,
                  // Then goes idle.
                  ASTAction::Idle,
                  // Afterwards we start executing go-to-def.
                  ASTAction::RunningAction,
                  // Then go idle.
                  ASTAction::Idle));
}

TEST_F(TUSchedulerTests, CommandLineErrors) {
  // We should see errors from command-line parsing inside the main file.
  CDB.ExtraClangFlags = {"-fsome-unknown-flag"};

  // (!) 'Ready' must live longer than TUScheduler.
  Notification Ready;

  TUScheduler S(CDB, optsForTest(), captureDiags());
  std::vector<Diag> Diagnostics;
  updateWithDiags(S, testPath("foo.cpp"), "void test() {}",
                  WantDiagnostics::Yes, [&](std::vector<Diag> D) {
                    Diagnostics = std::move(D);
                    Ready.notify();
                  });
  Ready.wait();

  EXPECT_THAT(
      Diagnostics,
      ElementsAre(AllOf(
          Field(&Diag::ID, Eq(diag::err_drv_unknown_argument)),
          Field(&Diag::Name, Eq("drv_unknown_argument")),
          Field(&Diag::Message, "unknown argument: '-fsome-unknown-flag'"))));
}

TEST_F(TUSchedulerTests, CommandLineWarnings) {
  // We should not see warnings from command-line parsing.
  CDB.ExtraClangFlags = {"-Wsome-unknown-warning"};

  // (!) 'Ready' must live longer than TUScheduler.
  Notification Ready;

  TUScheduler S(CDB, optsForTest(), captureDiags());
  std::vector<Diag> Diagnostics;
  updateWithDiags(S, testPath("foo.cpp"), "void test() {}",
                  WantDiagnostics::Yes, [&](std::vector<Diag> D) {
                    Diagnostics = std::move(D);
                    Ready.notify();
                  });
  Ready.wait();

  EXPECT_THAT(Diagnostics, IsEmpty());
}

TEST(DebouncePolicy, Compute) {
  namespace c = std::chrono;
  std::vector<DebouncePolicy::clock::duration> History = {
      c::seconds(0),
      c::seconds(5),
      c::seconds(10),
      c::seconds(20),
  };
  DebouncePolicy Policy;
  Policy.Min = c::seconds(3);
  Policy.Max = c::seconds(25);
  // Call Policy.compute(History) and return seconds as a float.
  auto Compute = [&](llvm::ArrayRef<DebouncePolicy::clock::duration> History) {
    using FloatingSeconds = c::duration<float, c::seconds::period>;
    return static_cast<float>(Policy.compute(History) / FloatingSeconds(1));
  };
  EXPECT_NEAR(10, Compute(History), 0.01) << "(upper) median = 10";
  Policy.RebuildRatio = 1.5;
  EXPECT_NEAR(15, Compute(History), 0.01) << "median = 10, ratio = 1.5";
  Policy.RebuildRatio = 3;
  EXPECT_NEAR(25, Compute(History), 0.01) << "constrained by max";
  Policy.RebuildRatio = 0;
  EXPECT_NEAR(3, Compute(History), 0.01) << "constrained by min";
  EXPECT_NEAR(25, Compute({}), 0.01) << "no history -> max";
}

} // namespace
} // namespace clangd
} // namespace clang
