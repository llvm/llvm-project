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

using ::testing::Pair;
using ::testing::Pointee;

void ignoreUpdate(llvm::Optional<std::vector<DiagWithFixIts>>) {}
void ignoreError(llvm::Error Err) {
  handleAllErrors(std::move(Err), [](const llvm::ErrorInfoBase &) {});
}

class TUSchedulerTests : public ::testing::Test {
protected:
  ParseInputs getInputs(PathRef File, std::string Contents) {
    return ParseInputs{*CDB.getCompileCommand(File), buildTestFS(Files),
                       std::move(Contents)};
  }

  llvm::StringMap<std::string> Files;

private:
  MockCompilationDatabase CDB;
};

TEST_F(TUSchedulerTests, MissingFiles) {
  TUScheduler S(getDefaultAsyncThreadsCount(),
                /*StorePreamblesInMemory=*/true,
                /*ASTParsedCallback=*/nullptr,
                /*UpdateDebounce=*/std::chrono::steady_clock::duration::zero());

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
        /*ASTParsedCallback=*/nullptr,
        /*UpdateDebounce=*/std::chrono::steady_clock::duration::zero());
    auto Path = testPath("foo.cpp");
    S.update(Path, getInputs(Path, ""), WantDiagnostics::Yes,
             [&](std::vector<DiagWithFixIts>) { Ready.wait(); });

    S.update(Path, getInputs(Path, "request diags"), WantDiagnostics::Yes,
             [&](std::vector<DiagWithFixIts> Diags) { ++CallbackCount; });
    S.update(Path, getInputs(Path, "auto (clobbered)"), WantDiagnostics::Auto,
             [&](std::vector<DiagWithFixIts> Diags) {
               ADD_FAILURE() << "auto should have been cancelled by auto";
             });
    S.update(Path, getInputs(Path, "request no diags"), WantDiagnostics::No,
             [&](std::vector<DiagWithFixIts> Diags) {
               ADD_FAILURE() << "no diags should not be called back";
             });
    S.update(Path, getInputs(Path, "auto (produces)"), WantDiagnostics::Auto,
             [&](std::vector<DiagWithFixIts> Diags) { ++CallbackCount; });
    Ready.notify();
  }
  EXPECT_EQ(2, CallbackCount);
}

TEST_F(TUSchedulerTests, Debounce) {
  std::atomic<int> CallbackCount(0);
  {
    TUScheduler S(getDefaultAsyncThreadsCount(),
                  /*StorePreamblesInMemory=*/true,
                  /*ASTParsedCallback=*/nullptr,
                  /*UpdateDebounce=*/std::chrono::milliseconds(50));
    auto Path = testPath("foo.cpp");
    S.update(Path, getInputs(Path, "auto (debounced)"), WantDiagnostics::Auto,
             [&](std::vector<DiagWithFixIts> Diags) {
               ADD_FAILURE() << "auto should have been debounced and canceled";
             });
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    S.update(Path, getInputs(Path, "auto (timed out)"), WantDiagnostics::Auto,
             [&](std::vector<DiagWithFixIts> Diags) { ++CallbackCount; });
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    S.update(Path, getInputs(Path, "auto (shut down)"), WantDiagnostics::Auto,
             [&](std::vector<DiagWithFixIts> Diags) { ++CallbackCount; });
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
                  /*ASTParsedCallback=*/nullptr,
                  /*UpdateDebounce=*/std::chrono::milliseconds(50));

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
                   [Nonce, &Mut, &TotalUpdates](
                       llvm::Optional<std::vector<DiagWithFixIts>> Diags) {
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
          S.runWithPreamble(
              "CheckPreamble", File,
              [Inputs, Nonce, &Mut, &TotalPreambleReads](
                  llvm::Expected<InputsAndPreamble> Preamble) {
                EXPECT_THAT(Context::current().get(NonceKey), Pointee(Nonce));

                ASSERT_TRUE((bool)Preamble);
                EXPECT_EQ(Preamble->Inputs.FS, Inputs.FS);
                EXPECT_EQ(Preamble->Inputs.Contents, Inputs.Contents);

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

} // namespace clangd
} // namespace clang
