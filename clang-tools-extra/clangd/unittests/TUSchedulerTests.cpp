//===-- TUSchedulerTests.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "ClangdServer.h"
#include "Diagnostics.h"
#include "GlobalCompilationDatabase.h"
#include "Matchers.h"
#include "ParsedAST.h"
#include "Preamble.h"
#include "TUScheduler.h"
#include "TestFS.h"
#include "TestIndex.h"
#include "support/Cancellation.h"
#include "support/Context.h"
#include "support/Path.h"
#include "support/TestTracer.h"
#include "support/Threading.h"
#include "support/ThreadsafeFS.h"
#include "clang/Basic/DiagnosticDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace clang {
namespace clangd {
namespace {

using ::testing::AllOf;
using ::testing::AnyOf;
using ::testing::Contains;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Field;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::Pair;
using ::testing::Pointee;
using ::testing::SizeIs;
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

// Simple ContextProvider to verify the provider is invoked & contexts are used.
static Key<std::string> BoundPath;
Context bindPath(PathRef F) {
  return Context::current().derive(BoundPath, F.str());
}
llvm::StringRef boundPath() {
  const std::string *V = Context::current().get(BoundPath);
  return V ? *V : llvm::StringRef("");
}

TUScheduler::Options optsForTest() {
  TUScheduler::Options Opts(ClangdServer::optsForTest());
  Opts.ContextProvider = bindPath;
  return Opts;
}

class TUSchedulerTests : public ::testing::Test {
protected:
  ParseInputs getInputs(PathRef File, std::string Contents) {
    ParseInputs Inputs;
    Inputs.CompileCommand = *CDB.getCompileCommand(File);
    Inputs.TFS = &FS;
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
        reportDiagnostics(File, *AST.getDiagnostics(), Publish);
      }

      void onFailedAST(PathRef File, llvm::StringRef Version,
                       std::vector<Diag> Diags, PublishFn Publish) override {
        reportDiagnostics(File, Diags, Publish);
      }

    private:
      void reportDiagnostics(PathRef File, llvm::ArrayRef<Diag> Diags,
                             PublishFn Publish) {
        auto *D = Context::current().get(DiagsCallbackKey);
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

  MockFS FS;
  MockCompilationDatabase CDB;
};

Key<llvm::unique_function<void(PathRef File, std::vector<Diag>)>>
    TUSchedulerTests::DiagsCallbackKey;

TEST_F(TUSchedulerTests, MissingFiles) {
  TUScheduler S(CDB, optsForTest());

  auto Added = testPath("added.cpp");
  FS.Files[Added] = "x";

  auto Missing = testPath("missing.cpp");
  FS.Files[Missing] = "";

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
  auto Opts = optsForTest();
  Opts.UpdateDebounce = DebouncePolicy::fixed(std::chrono::milliseconds(500));
  TUScheduler S(CDB, Opts, captureDiags());
  auto Path = testPath("foo.cpp");
  // Issue a write that's going to be debounced away.
  updateWithDiags(S, Path, "auto (debounced)", WantDiagnostics::Auto,
                  [&](std::vector<Diag>) {
                    ADD_FAILURE()
                        << "auto should have been debounced and canceled";
                  });
  // Sleep a bit to verify that it's really debounce that's holding diagnostics.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Issue another write, this time we'll wait for its diagnostics.
  Notification N;
  updateWithDiags(S, Path, "auto (timed out)", WantDiagnostics::Auto,
                  [&](std::vector<Diag>) { N.notify(); });
  EXPECT_TRUE(N.wait(timeoutSeconds(5)));

  // Once we start shutting down the TUScheduler, this one becomes a dead write.
  updateWithDiags(S, Path, "auto (discarded)", WantDiagnostics::Auto,
                  [&](std::vector<Diag>) {
                    ADD_FAILURE()
                        << "auto should have been discarded (dead write)";
                  });
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
  std::vector<StringRef> DiagsSeen, ReadsSeen, ReadsCanceled;
  {
    Notification Proceed; // Ensure we schedule everything.
    TUScheduler S(CDB, optsForTest(), captureDiags());
    auto Path = testPath("foo.cpp");
    // Helper to schedule a named update and return a function to cancel it.
    auto Update = [&](StringRef ID) -> Canceler {
      auto T = cancelableTask();
      WithContext C(std::move(T.first));
      updateWithDiags(
          S, Path, ("//" + ID).str(), WantDiagnostics::Yes,
          [&, ID](std::vector<Diag> Diags) { DiagsSeen.push_back(ID); });
      return std::move(T.second);
    };
    // Helper to schedule a named read and return a function to cancel it.
    auto Read = [&](StringRef ID) -> Canceler {
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

// We don't invalidate requests for updates that don't change the file content.
// These are mostly "refresh this file" events synthesized inside clangd itself.
// (Usually the AST rebuild is elided after verifying that all inputs are
// unchanged, but invalidation decisions happen earlier and so independently).
// See https://github.com/clangd/clangd/issues/620
TEST_F(TUSchedulerTests, InvalidationUnchanged) {
  auto Path = testPath("foo.cpp");
  TUScheduler S(CDB, optsForTest(), captureDiags());
  std::atomic<int> Actions(0);

  Notification Start;
  updateWithDiags(S, Path, "a", WantDiagnostics::Yes, [&](std::vector<Diag>) {
    Start.wait();
  });
  S.runWithAST(
      "invalidatable", Path,
      [&](llvm::Expected<InputsAndAST> AST) {
        ++Actions;
        EXPECT_TRUE(bool(AST))
            << "Should not invalidate based on an update with same content: "
            << llvm::toString(AST.takeError());
      },
      TUScheduler::InvalidateOnUpdate);
  updateWithDiags(S, Path, "a", WantDiagnostics::Yes, [&](std::vector<Diag>) {
    ADD_FAILURE() << "Shouldn't build, identical to previous";
  });
  Start.notify();
  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));

  EXPECT_EQ(1, Actions.load()) << "All actions should run";
}

TEST_F(TUSchedulerTests, ManyUpdates) {
  const int FilesCount = 3;
  const int UpdatesPerFile = 10;

  std::mutex Mut;
  int TotalASTReads = 0;
  int TotalPreambleReads = 0;
  int TotalUpdates = 0;
  llvm::StringMap<int> LatestDiagVersion;

  // Run TUScheduler and collect some stats.
  {
    auto Opts = optsForTest();
    Opts.UpdateDebounce = DebouncePolicy::fixed(std::chrono::milliseconds(50));
    TUScheduler S(CDB, Opts, captureDiags());

    std::vector<std::string> Files;
    for (int I = 0; I < FilesCount; ++I) {
      std::string Name = "foo" + std::to_string(I) + ".cpp";
      Files.push_back(testPath(Name));
      this->FS.Files[Files.back()] = "";
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
          Inputs.Version = std::to_string(UpdateI);
          updateWithDiags(
              S, File, Inputs, WantDiagnostics::Auto,
              [File, Nonce, Version(Inputs.Version), &Mut, &TotalUpdates,
               &LatestDiagVersion](std::vector<Diag>) {
                EXPECT_THAT(Context::current().get(NonceKey), Pointee(Nonce));
                EXPECT_EQ(File, boundPath());

                std::lock_guard<std::mutex> Lock(Mut);
                ++TotalUpdates;
                EXPECT_EQ(File, *TUScheduler::getFileBeingProcessedInContext());
                // Make sure Diags are for a newer version.
                auto It = LatestDiagVersion.try_emplace(File, -1);
                const int PrevVersion = It.first->second;
                int CurVersion;
                ASSERT_TRUE(llvm::to_integer(Version, CurVersion, 10));
                EXPECT_LT(PrevVersion, CurVersion);
                It.first->getValue() = CurVersion;
              });
        }
        {
          WithContextValue WithNonce(NonceKey, ++Nonce);
          S.runWithAST(
              "CheckAST", File,
              [File, Inputs, Nonce, &Mut,
               &TotalASTReads](Expected<InputsAndAST> AST) {
                EXPECT_THAT(Context::current().get(NonceKey), Pointee(Nonce));
                EXPECT_EQ(File, boundPath());

                ASSERT_TRUE((bool)AST);
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
                EXPECT_EQ(File, boundPath());

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
  // Updates might get coalesced in preamble thread and result in dropping
  // diagnostics for intermediate snapshots.
  EXPECT_GE(TotalUpdates, FilesCount);
  EXPECT_LE(TotalUpdates, FilesCount * UpdatesPerFile);
  // We should receive diags for last update.
  for (const auto &Entry : LatestDiagVersion)
    EXPECT_EQ(Entry.second, UpdatesPerFile - 1);
  EXPECT_EQ(TotalASTReads, FilesCount * UpdatesPerFile);
  EXPECT_EQ(TotalPreambleReads, FilesCount * UpdatesPerFile);
}

TEST_F(TUSchedulerTests, EvictedAST) {
  std::atomic<int> BuiltASTCounter(0);
  auto Opts = optsForTest();
  Opts.AsyncThreadsCount = 1;
  Opts.RetentionPolicy.MaxRetainedASTs = 2;
  trace::TestTracer Tracer;
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

  EXPECT_THAT(Tracer.takeMetric("ast_access_diag", "hit"), SizeIs(0));
  EXPECT_THAT(Tracer.takeMetric("ast_access_diag", "miss"), SizeIs(0));
  // Build one file in advance. We will not access it later, so it will be the
  // one that the cache will evict.
  updateWithCallback(S, Foo, SourceContents, WantDiagnostics::Yes,
                     [&BuiltASTCounter]() { ++BuiltASTCounter; });
  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
  ASSERT_EQ(BuiltASTCounter.load(), 1);
  EXPECT_THAT(Tracer.takeMetric("ast_access_diag", "hit"), SizeIs(0));
  EXPECT_THAT(Tracer.takeMetric("ast_access_diag", "miss"), SizeIs(1));

  // Build two more files. Since we can retain only 2 ASTs, these should be
  // the ones we see in the cache later.
  updateWithCallback(S, Bar, SourceContents, WantDiagnostics::Yes,
                     [&BuiltASTCounter]() { ++BuiltASTCounter; });
  updateWithCallback(S, Baz, SourceContents, WantDiagnostics::Yes,
                     [&BuiltASTCounter]() { ++BuiltASTCounter; });
  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
  ASSERT_EQ(BuiltASTCounter.load(), 3);
  EXPECT_THAT(Tracer.takeMetric("ast_access_diag", "hit"), SizeIs(0));
  EXPECT_THAT(Tracer.takeMetric("ast_access_diag", "miss"), SizeIs(2));

  // Check only the last two ASTs are retained.
  ASSERT_THAT(S.getFilesWithCachedAST(), UnorderedElementsAre(Bar, Baz));

  // Access the old file again.
  updateWithCallback(S, Foo, OtherSourceContents, WantDiagnostics::Yes,
                     [&BuiltASTCounter]() { ++BuiltASTCounter; });
  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
  ASSERT_EQ(BuiltASTCounter.load(), 4);
  EXPECT_THAT(Tracer.takeMetric("ast_access_diag", "hit"), SizeIs(0));
  EXPECT_THAT(Tracer.takeMetric("ast_access_diag", "miss"), SizeIs(1));

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

  FS.Files[Header] = "void foo()";
  FS.Timestamps[Header] = time_t(0);
  auto *WithPreamble = R"cpp(
    #include "foo.h"
    int main() {}
  )cpp";
  auto *WithEmptyPreamble = R"cpp(int main() {})cpp";
  S.update(Foo, getInputs(Foo, WithPreamble), WantDiagnostics::Auto);
  S.runWithPreamble(
      "getNonEmptyPreamble", Foo, TUScheduler::Stale,
      [&](Expected<InputsAndPreamble> Preamble) {
        // We expect to get a non-empty preamble.
        EXPECT_GT(
            cantFail(std::move(Preamble)).Preamble->Preamble.getBounds().Size,
            0u);
      });
  // Wait while the preamble is being built.
  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));

  // Update the file which results in an empty preamble.
  S.update(Foo, getInputs(Foo, WithEmptyPreamble), WantDiagnostics::Auto);
  // Wait while the preamble is being built.
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

TEST_F(TUSchedulerTests, ASTSignalsSmokeTests) {
  TUScheduler S(CDB, optsForTest());
  auto Foo = testPath("foo.cpp");
  auto Header = testPath("foo.h");

  FS.Files[Header] = "namespace tar { int foo(); }";
  const char *Contents = R"cpp(
  #include "foo.h"
  namespace ns {
  int func() {
    return tar::foo());
  }
  } // namespace ns
  )cpp";
  // Update the file which results in an empty preamble.
  S.update(Foo, getInputs(Foo, Contents), WantDiagnostics::Yes);
  // Wait while the preamble is being built.
  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
  Notification TaskRun;
  S.runWithPreamble(
      "ASTSignals", Foo, TUScheduler::Stale,
      [&](Expected<InputsAndPreamble> IP) {
        ASSERT_FALSE(!IP);
        std::vector<std::pair<StringRef, int>> NS;
        for (const auto &P : IP->Signals->RelatedNamespaces)
          NS.emplace_back(P.getKey(), P.getValue());
        EXPECT_THAT(NS,
                    UnorderedElementsAre(Pair("ns::", 1), Pair("tar::", 1)));

        std::vector<std::pair<SymbolID, int>> Sym;
        for (const auto &P : IP->Signals->ReferencedSymbols)
          Sym.emplace_back(P.getFirst(), P.getSecond());
        EXPECT_THAT(Sym, UnorderedElementsAre(Pair(ns("tar").ID, 1),
                                              Pair(ns("ns").ID, 1),
                                              Pair(func("tar::foo").ID, 1),
                                              Pair(func("ns::func").ID, 1)));
        TaskRun.notify();
      });
  TaskRun.wait();
}

TEST_F(TUSchedulerTests, RunWaitsForPreamble) {
  // Testing strategy: we update the file and schedule a few preamble reads at
  // the same time. All reads should get the same non-null preamble.
  TUScheduler S(CDB, optsForTest());
  auto Foo = testPath("foo.cpp");
  auto *NonEmptyPreamble = R"cpp(
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

  FS.Files[Header] = "int a;";
  FS.Timestamps[Header] = time_t(0);

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
  FS.Timestamps[Header] = time_t(1);
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

// We rebuild if a completely missing header exists, but not if one is added
// on a higher-priority include path entry (for performance).
// (Previously we wouldn't automatically rebuild when files were added).
TEST_F(TUSchedulerTests, MissingHeader) {
  CDB.ExtraClangFlags.push_back("-I" + testPath("a"));
  CDB.ExtraClangFlags.push_back("-I" + testPath("b"));
  // Force both directories to exist so they don't get pruned.
  FS.Files.try_emplace("a/__unused__");
  FS.Files.try_emplace("b/__unused__");
  TUScheduler S(CDB, optsForTest(), captureDiags());

  auto Source = testPath("foo.cpp");
  auto HeaderA = testPath("a/foo.h");
  auto HeaderB = testPath("b/foo.h");

  auto *SourceContents = R"cpp(
      #include "foo.h"
      int c = b;
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
                                      "use of undeclared identifier 'b'")));
      });
  S.blockUntilIdle(timeoutSeconds(10));

  FS.Files[HeaderB] = "int b;";
  FS.Timestamps[HeaderB] = time_t(1);

  // The addition of the missing header file triggers a rebuild, no errors.
  updateWithDiags(S, Source, Inputs, WantDiagnostics::Yes,
                  [&DiagCount](std::vector<Diag> Diags) {
                    ++DiagCount;
                    EXPECT_THAT(Diags, IsEmpty());
                  });

  // Ensure previous assertions are done before we touch the FS again.
  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
  // Add the high-priority header file, which should reintroduce the error.
  FS.Files[HeaderA] = "int a;";
  FS.Timestamps[HeaderA] = time_t(1);

  // This isn't detected: we don't stat a/foo.h to validate the preamble.
  updateWithDiags(S, Source, Inputs, WantDiagnostics::Yes,
                  [&DiagCount](std::vector<Diag> Diags) {
                    ++DiagCount;
                    ADD_FAILURE()
                        << "Didn't expect new diagnostics when adding a/foo.h";
                  });

  // Forcing the reload should cause a rebuild.
  Inputs.ForceRebuild = true;
  updateWithDiags(
      S, Source, Inputs, WantDiagnostics::Yes,
      [&DiagCount](std::vector<Diag> Diags) {
        ++DiagCount;
        ElementsAre(Field(&Diag::Message, "use of undeclared identifier 'b'"));
      });

  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
  EXPECT_EQ(DiagCount, 3U);
}

TEST_F(TUSchedulerTests, NoChangeDiags) {
  trace::TestTracer Tracer;
  TUScheduler S(CDB, optsForTest(), captureDiags());

  auto FooCpp = testPath("foo.cpp");
  const auto *Contents = "int a; int b;";

  EXPECT_THAT(Tracer.takeMetric("ast_access_read", "hit"), SizeIs(0));
  EXPECT_THAT(Tracer.takeMetric("ast_access_read", "miss"), SizeIs(0));
  EXPECT_THAT(Tracer.takeMetric("ast_access_diag", "hit"), SizeIs(0));
  EXPECT_THAT(Tracer.takeMetric("ast_access_diag", "miss"), SizeIs(0));
  updateWithDiags(
      S, FooCpp, Contents, WantDiagnostics::No,
      [](std::vector<Diag>) { ADD_FAILURE() << "Should not be called."; });
  S.runWithAST("touchAST", FooCpp, [](Expected<InputsAndAST> IA) {
    // Make sure the AST was actually built.
    cantFail(std::move(IA));
  });
  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
  EXPECT_THAT(Tracer.takeMetric("ast_access_read", "hit"), SizeIs(0));
  EXPECT_THAT(Tracer.takeMetric("ast_access_read", "miss"), SizeIs(1));

  // Even though the inputs didn't change and AST can be reused, we need to
  // report the diagnostics, as they were not reported previously.
  std::atomic<bool> SeenDiags(false);
  updateWithDiags(S, FooCpp, Contents, WantDiagnostics::Auto,
                  [&](std::vector<Diag>) { SeenDiags = true; });
  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
  ASSERT_TRUE(SeenDiags);
  EXPECT_THAT(Tracer.takeMetric("ast_access_diag", "hit"), SizeIs(1));
  EXPECT_THAT(Tracer.takeMetric("ast_access_diag", "miss"), SizeIs(0));

  // Subsequent request does not get any diagnostics callback because the same
  // diags have previously been reported and the inputs didn't change.
  updateWithDiags(
      S, FooCpp, Contents, WantDiagnostics::Auto,
      [&](std::vector<Diag>) { ADD_FAILURE() << "Should not be called."; });
  ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
}

TEST_F(TUSchedulerTests, Run) {
  for (bool Sync : {false, true}) {
    auto Opts = optsForTest();
    if (Sync)
      Opts.AsyncThreadsCount = 0;
    TUScheduler S(CDB, Opts);
    std::atomic<int> Counter(0);
    S.run("add 1", /*Path=*/"", [&] { ++Counter; });
    S.run("add 2", /*Path=*/"", [&] { Counter += 2; });
    ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
    EXPECT_EQ(Counter.load(), 3);

    Notification TaskRun;
    Key<int> TestKey;
    WithContextValue CtxWithKey(TestKey, 10);
    const char *Path = "somepath";
    S.run("props context", Path, [&] {
      EXPECT_EQ(Context::current().getExisting(TestKey), 10);
      EXPECT_EQ(Path, boundPath());
      TaskRun.notify();
    });
    TaskRun.wait();
  }
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
  MockFS FS;
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
                  // Afterwards it builds an AST for that preamble to publish
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
  DebouncePolicy::clock::duration History[] = {
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
    return c::duration_cast<c::duration<float, c::seconds::period>>(
               Policy.compute(History))
        .count();
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

TEST_F(TUSchedulerTests, AsyncPreambleThread) {
  // Blocks preamble thread while building preamble with \p BlockVersion until
  // \p N is notified.
  class BlockPreambleThread : public ParsingCallbacks {
  public:
    BlockPreambleThread(llvm::StringRef BlockVersion, Notification &N)
        : BlockVersion(BlockVersion), N(N) {}
    void onPreambleAST(PathRef Path, llvm::StringRef Version,
                       const CompilerInvocation &, ASTContext &Ctx,
                       Preprocessor &, const CanonicalIncludes &) override {
      if (Version == BlockVersion)
        N.wait();
    }

  private:
    llvm::StringRef BlockVersion;
    Notification &N;
  };

  static constexpr llvm::StringLiteral InputsV0 = "v0";
  static constexpr llvm::StringLiteral InputsV1 = "v1";
  Notification Ready;
  TUScheduler S(CDB, optsForTest(),
                std::make_unique<BlockPreambleThread>(InputsV1, Ready));

  Path File = testPath("foo.cpp");
  auto PI = getInputs(File, "");
  PI.Version = InputsV0.str();
  S.update(File, PI, WantDiagnostics::Auto);
  S.blockUntilIdle(timeoutSeconds(10));

  // Block preamble builds.
  PI.Version = InputsV1.str();
  // Issue second update which will block preamble thread.
  S.update(File, PI, WantDiagnostics::Auto);

  Notification RunASTAction;
  // Issue an AST read, which shouldn't be blocked and see latest version of the
  // file.
  S.runWithAST("test", File, [&](Expected<InputsAndAST> AST) {
    ASSERT_TRUE(bool(AST));
    // Make sure preamble is built with stale inputs, but AST was built using
    // new ones.
    EXPECT_THAT(AST->AST.preambleVersion(), InputsV0);
    EXPECT_THAT(AST->Inputs.Version, InputsV1.str());
    RunASTAction.notify();
  });
  RunASTAction.wait();
  Ready.notify();
}

TEST_F(TUSchedulerTests, OnlyPublishWhenPreambleIsBuilt) {
  struct PreamblePublishCounter : public ParsingCallbacks {
    PreamblePublishCounter(int &PreamblePublishCount)
        : PreamblePublishCount(PreamblePublishCount) {}
    void onPreamblePublished(PathRef File) override { ++PreamblePublishCount; }
    int &PreamblePublishCount;
  };

  int PreamblePublishCount = 0;
  TUScheduler S(CDB, optsForTest(),
                std::make_unique<PreamblePublishCounter>(PreamblePublishCount));

  Path File = testPath("foo.cpp");
  S.update(File, getInputs(File, ""), WantDiagnostics::Auto);
  S.blockUntilIdle(timeoutSeconds(10));
  EXPECT_EQ(PreamblePublishCount, 1);
  // Same contents, no publish.
  S.update(File, getInputs(File, ""), WantDiagnostics::Auto);
  S.blockUntilIdle(timeoutSeconds(10));
  EXPECT_EQ(PreamblePublishCount, 1);
  // New contents, should publish.
  S.update(File, getInputs(File, "#define FOO"), WantDiagnostics::Auto);
  S.blockUntilIdle(timeoutSeconds(10));
  EXPECT_EQ(PreamblePublishCount, 2);
}

// If a header file is missing from the CDB (or inferred using heuristics), and
// it's included by another open file, then we parse it using that files flags.
TEST_F(TUSchedulerTests, IncluderCache) {
  static std::string Main = testPath("main.cpp"), Main2 = testPath("main2.cpp"),
                     Main3 = testPath("main3.cpp"),
                     NoCmd = testPath("no_cmd.h"),
                     Unreliable = testPath("unreliable.h"),
                     OK = testPath("ok.h"),
                     NotIncluded = testPath("not_included.h");
  struct NoHeadersCDB : public GlobalCompilationDatabase {
    llvm::Optional<tooling::CompileCommand>
    getCompileCommand(PathRef File) const override {
      if (File == NoCmd || File == NotIncluded || FailAll)
        return llvm::None;
      auto Basic = getFallbackCommand(File);
      Basic.Heuristic.clear();
      if (File == Unreliable) {
        Basic.Heuristic = "not reliable";
      } else if (File == Main) {
        Basic.CommandLine.push_back("-DMAIN");
      } else if (File == Main2) {
        Basic.CommandLine.push_back("-DMAIN2");
      } else if (File == Main3) {
        Basic.CommandLine.push_back("-DMAIN3");
      }
      return Basic;
    }

    std::atomic<bool> FailAll{false};
  } CDB;
  TUScheduler S(CDB, optsForTest());
  auto GetFlags = [&](PathRef Header) {
    S.update(Header, getInputs(Header, ";"), WantDiagnostics::Yes);
    EXPECT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
    Notification CmdDone;
    tooling::CompileCommand Cmd;
    S.runWithPreamble("GetFlags", Header, TUScheduler::StaleOrAbsent,
                      [&](llvm::Expected<InputsAndPreamble> Inputs) {
                        ASSERT_FALSE(!Inputs) << Inputs.takeError();
                        Cmd = std::move(Inputs->Command);
                        CmdDone.notify();
                      });
    CmdDone.wait();
    EXPECT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
    return Cmd.CommandLine;
  };

  for (const auto &Path : {NoCmd, Unreliable, OK, NotIncluded})
    FS.Files[Path] = ";";

  // Initially these files have normal commands from the CDB.
  EXPECT_THAT(GetFlags(Main), Contains("-DMAIN")) << "sanity check";
  EXPECT_THAT(GetFlags(NoCmd), Not(Contains("-DMAIN"))) << "no includes yet";

  // Now make Main include the others, and some should pick up its flags.
  const char *AllIncludes = R"cpp(
    #include "no_cmd.h"
    #include "ok.h"
    #include "unreliable.h"
  )cpp";
  S.update(Main, getInputs(Main, AllIncludes), WantDiagnostics::Yes);
  EXPECT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
  EXPECT_THAT(GetFlags(NoCmd), Contains("-DMAIN"))
      << "Included from main file, has no own command";
  EXPECT_THAT(GetFlags(Unreliable), Contains("-DMAIN"))
      << "Included from main file, own command is heuristic";
  EXPECT_THAT(GetFlags(OK), Not(Contains("-DMAIN")))
      << "Included from main file, but own command is used";
  EXPECT_THAT(GetFlags(NotIncluded), Not(Contains("-DMAIN")))
      << "Not included from main file";

  // Open another file - it won't overwrite the associations with Main.
  std::string SomeIncludes = R"cpp(
    #include "no_cmd.h"
    #include "not_included.h"
  )cpp";
  S.update(Main2, getInputs(Main2, SomeIncludes), WantDiagnostics::Yes);
  EXPECT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
  EXPECT_THAT(GetFlags(NoCmd),
              AllOf(Contains("-DMAIN"), Not(Contains("-DMAIN2"))))
      << "mainfile association is stable";
  EXPECT_THAT(GetFlags(NotIncluded),
              AllOf(Contains("-DMAIN2"), Not(Contains("-DMAIN"))))
      << "new headers are associated with new mainfile";

  // Remove includes from main - this marks the associations as invalid but
  // doesn't actually remove them until another preamble claims them.
  S.update(Main, getInputs(Main, ""), WantDiagnostics::Yes);
  EXPECT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
  EXPECT_THAT(GetFlags(NoCmd),
              AllOf(Contains("-DMAIN"), Not(Contains("-DMAIN2"))))
      << "mainfile association not updated yet!";

  // Open yet another file - this time it claims the associations.
  S.update(Main3, getInputs(Main3, SomeIncludes), WantDiagnostics::Yes);
  EXPECT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
  EXPECT_THAT(GetFlags(NoCmd), Contains("-DMAIN3"))
      << "association invalidated and then claimed by main3";
  EXPECT_THAT(GetFlags(Unreliable), Contains("-DMAIN"))
      << "association invalidated but not reclaimed";
  EXPECT_THAT(GetFlags(NotIncluded), Contains("-DMAIN2"))
      << "association still valid";

  // Delete the file from CDB, it should invalidate the associations.
  CDB.FailAll = true;
  EXPECT_THAT(GetFlags(NoCmd), Not(Contains("-DMAIN3")))
      << "association should've been invalidated.";
  // Also run update for Main3 to invalidate the preeamble to make sure next
  // update populates include cache associations.
  S.update(Main3, getInputs(Main3, SomeIncludes), WantDiagnostics::Yes);
  EXPECT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
  // Re-add the file and make sure nothing crashes.
  CDB.FailAll = false;
  S.update(Main3, getInputs(Main3, SomeIncludes), WantDiagnostics::Yes);
  EXPECT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
  EXPECT_THAT(GetFlags(NoCmd), Contains("-DMAIN3"))
      << "association invalidated and then claimed by main3";
}

TEST_F(TUSchedulerTests, PreservesLastActiveFile) {
  for (bool Sync : {false, true}) {
    auto Opts = optsForTest();
    if (Sync)
      Opts.AsyncThreadsCount = 0;
    TUScheduler S(CDB, Opts);

    auto CheckNoFileActionsSeesLastActiveFile =
        [&](llvm::StringRef LastActiveFile) {
          ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
          std::atomic<int> Counter(0);
          // We only check for run and runQuick as runWithAST and
          // runWithPreamble is always bound to a file.
          S.run("run-UsesLastActiveFile", /*Path=*/"", [&] {
            ++Counter;
            EXPECT_EQ(LastActiveFile, boundPath());
          });
          S.runQuick("runQuick-UsesLastActiveFile", /*Path=*/"", [&] {
            ++Counter;
            EXPECT_EQ(LastActiveFile, boundPath());
          });
          ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));
          EXPECT_EQ(2, Counter.load());
        };

    // Check that we see no file initially
    CheckNoFileActionsSeesLastActiveFile("");

    // Now check that every action scheduled with a particular file changes the
    // LastActiveFile.
    auto Path = testPath("run.cc");
    S.run(Path, Path, [] {});
    CheckNoFileActionsSeesLastActiveFile(Path);

    Path = testPath("runQuick.cc");
    S.runQuick(Path, Path, [] {});
    CheckNoFileActionsSeesLastActiveFile(Path);

    Path = testPath("runWithAST.cc");
    S.update(Path, getInputs(Path, ""), WantDiagnostics::No);
    S.runWithAST(Path, Path, [](llvm::Expected<InputsAndAST> Inp) {
      EXPECT_TRUE(bool(Inp));
    });
    CheckNoFileActionsSeesLastActiveFile(Path);

    Path = testPath("runWithPreamble.cc");
    S.update(Path, getInputs(Path, ""), WantDiagnostics::No);
    S.runWithPreamble(
        Path, Path, TUScheduler::Stale,
        [](llvm::Expected<InputsAndPreamble> Inp) { EXPECT_TRUE(bool(Inp)); });
    CheckNoFileActionsSeesLastActiveFile(Path);

    Path = testPath("update.cc");
    S.update(Path, getInputs(Path, ""), WantDiagnostics::No);
    CheckNoFileActionsSeesLastActiveFile(Path);

    // An update with the same contents should not change LastActiveFile.
    auto LastActive = Path;
    Path = testPath("runWithAST.cc");
    S.update(Path, getInputs(Path, ""), WantDiagnostics::No);
    CheckNoFileActionsSeesLastActiveFile(LastActive);
  }
}

TEST_F(TUSchedulerTests, PreambleThrottle) {
  const int NumRequests = 4;
  // Silly throttler that waits for 4 requests, and services them in reverse.
  // Doesn't honor cancellation but records it.
  struct : public PreambleThrottler {
    std::mutex Mu;
    std::vector<std::string> Acquires;
    std::vector<RequestID> Releases;
    llvm::DenseMap<RequestID, Callback> Callbacks;
    // If set, the notification is signalled after acquiring the specified ID.
    llvm::Optional<std::pair<RequestID, Notification *>> Notify;

    RequestID acquire(llvm::StringRef Filename, Callback CB) override {
      RequestID ID;
      Callback Invoke;
      {
        std::lock_guard<std::mutex> Lock(Mu);
        ID = Acquires.size();
        Acquires.emplace_back(Filename);
        // If we're full, satisfy this request immediately.
        if (Acquires.size() == NumRequests) {
          Invoke = std::move(CB);
        } else {
          Callbacks.try_emplace(ID, std::move(CB));
        }
      }
      if (Invoke)
        Invoke();
      {
        std::lock_guard<std::mutex> Lock(Mu);
        if (Notify && ID == Notify->first) {
          Notify->second->notify();
          Notify.reset();
        }
      }
      return ID;
    }

    void release(RequestID ID) override {
      Callback SatisfyNext;
      {
        std::lock_guard<std::mutex> Lock(Mu);
        Releases.push_back(ID);
        if (ID > 0 && Acquires.size() == NumRequests)
          SatisfyNext = std::move(Callbacks[ID - 1]);
      }
      if (SatisfyNext)
        SatisfyNext();
    }

    void reset() {
      Acquires.clear();
      Releases.clear();
      Callbacks.clear();
    }
  } Throttler;

  struct CaptureBuiltFilenames : public ParsingCallbacks {
    std::vector<std::string> &Filenames;
    CaptureBuiltFilenames(std::vector<std::string> &Filenames)
        : Filenames(Filenames) {}
    void onPreambleAST(PathRef Path, llvm::StringRef Version,
                       const CompilerInvocation &CI, ASTContext &Ctx,
                       Preprocessor &PP, const CanonicalIncludes &) override {
      // Deliberately no synchronization.
      // The PreambleThrottler should serialize these calls, if not then tsan
      // will find a bug here.
      Filenames.emplace_back(Path);
    }
  };

  auto Opts = optsForTest();
  Opts.AsyncThreadsCount = 2 * NumRequests; // throttler is the bottleneck
  Opts.PreambleThrottler = &Throttler;

  std::vector<std::string> Filenames;

  {
    std::vector<std::string> BuiltFilenames;
    TUScheduler S(CDB, Opts,
                  std::make_unique<CaptureBuiltFilenames>(BuiltFilenames));
    for (unsigned I = 0; I < NumRequests; ++I) {
      auto Path = testPath(std::to_string(I) + ".cc");
      Filenames.push_back(Path);
      S.update(Path, getInputs(Path, ""), WantDiagnostics::Yes);
    }
    ASSERT_TRUE(S.blockUntilIdle(timeoutSeconds(10)));

    // The throttler saw all files, and we built them.
    EXPECT_THAT(Throttler.Acquires,
                testing::UnorderedElementsAreArray(Filenames));
    EXPECT_THAT(BuiltFilenames,
                testing::UnorderedElementsAreArray(Filenames));
    // We built the files in reverse order that the throttler saw them.
    EXPECT_THAT(BuiltFilenames,
                testing::ElementsAreArray(Throttler.Acquires.rbegin(),
                                          Throttler.Acquires.rend()));
    // Resources for each file were correctly released.
    EXPECT_THAT(Throttler.Releases, ElementsAre(3, 2, 1, 0));
  }

  Throttler.reset();

  // This time, enqueue 2 files, then cancel one of them while still waiting.
  // Finally shut down the server. Observe that everything gets cleaned up.
  Notification AfterAcquire2;
  Notification AfterFinishA;
  Throttler.Notify = {1, &AfterAcquire2};
  std::vector<std::string> BuiltFilenames;
  auto A = testPath("a.cc");
  auto B = testPath("b.cc");
  Filenames = {A, B};
  {
    TUScheduler S(CDB, Opts,
                  std::make_unique<CaptureBuiltFilenames>(BuiltFilenames));
    updateWithCallback(S, A, getInputs(A, ""), WantDiagnostics::Yes,
                       [&] { AfterFinishA.notify(); });
    S.update(B, getInputs(B, ""), WantDiagnostics::Yes);
    AfterAcquire2.wait();

    // The throttler saw all files, but we built none.
    EXPECT_THAT(Throttler.Acquires,
                testing::UnorderedElementsAreArray(Filenames));
    EXPECT_THAT(BuiltFilenames, testing::IsEmpty());
    // We haven't released anything yet, we're still waiting.
    EXPECT_THAT(Throttler.Releases, testing::IsEmpty());

    // FIXME: This is flaky, becaues the request can be destroyed after shutdown
    // if it hasn't been dequeued yet (stop() resets NextRequest).
#if 0
    // Now close file A, which will shut down its AST worker.
    S.remove(A);
    // Request is destroyed after the queue shutdown, so release() has happened.
    AfterFinishA.wait();
    // We still didn't build anything.
    EXPECT_THAT(BuiltFilenames, testing::IsEmpty());
    // But we've cancelled the request to build A (not sure which its ID is).
    EXPECT_THAT(Throttler.Releases, ElementsAre(AnyOf(1, 0)));
#endif

    // Now shut down the TU Scheduler.
  }
  // The throttler saw all files, but we built none.
  EXPECT_THAT(Throttler.Acquires,
              testing::UnorderedElementsAreArray(Filenames));
  EXPECT_THAT(BuiltFilenames, testing::IsEmpty());
  // We gave up waiting and everything got released (in some order).
  EXPECT_THAT(Throttler.Releases, UnorderedElementsAre(1, 0));
}

} // namespace
} // namespace clangd
} // namespace clang
