//===- CompilationTest.cpp - Job execution tests ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Compilation.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace clang::driver;

namespace {
class CompilationTest : public testing::Test {
protected:
  struct Diagnostics : DiagnosticConsumer {
    std::string Message;
    void HandleDiagnostic(DiagnosticsEngine::Level Level,
                          const Diagnostic &Info) override {
      DiagnosticConsumer::HandleDiagnostic(Level, Info);
      llvm::SmallString<128> Text;
      Info.FormatDiagnostic(Text);
      Message = std::string(Text);
    }
  } Client;
  DiagnosticOptions DiagOpts;
  DiagnosticsEngine Diags{DiagnosticIDs::create(), DiagOpts, &Client, false};
  Driver D{"/clang", "wasm32-unknown-unknown", Diags};
  llvm::SmallString<128> Directory;
  llvm::SmallString<128> Output;
  llvm::SmallString<128> MissingLinker;
  std::unique_ptr<Compilation> C;

  void SetUp() override {
    ASSERT_FALSE(
        llvm::sys::fs::createUniqueDirectory("clang-executor", Directory));
    Output = Directory;
    llvm::sys::path::append(Output, "program.wasm");
    MissingLinker = Directory;
    llvm::sys::path::append(MissingLinker, "missing-linker");
    D.setCheckInputsExist(false);
    C.reset(D.BuildCompilation(
        {"clang", "-nostdlib", "input.o", "-o", Output.c_str()}));
    ASSERT_TRUE(C);
    ASSERT_FALSE(C->containsError());
    ASSERT_EQ(C->getJobs().size(), 1u);
    C->getJobs().begin()->replaceExecutable(MissingLinker.c_str());
  }

  void TearDown() override {
    C.reset();
    llvm::sys::fs::remove_directories(Directory);
  }
};

TEST_F(CompilationTest, DefaultExecution) {
  const Command *Failed = nullptr;
  EXPECT_EQ(C->ExecuteCommand(*C->getJobs().begin(), Failed), 1);
  EXPECT_EQ(Failed, &*C->getJobs().begin());
  EXPECT_EQ(Diags.getNumErrors(), 1u);
}

TEST_F(CompilationTest, UnhandledJobUsesDefaultExecution) {
  unsigned Calls = 0;
  C->setJobExecutor([&](const Command &, auto, auto &, bool &) {
    ++Calls;
    return std::optional<int>();
  });
  const Command *Failed = nullptr;
  EXPECT_EQ(C->ExecuteCommand(*C->getJobs().begin(), Failed), 1);
  EXPECT_EQ(Calls, 1u);
  EXPECT_EQ(Failed, &*C->getJobs().begin());
  EXPECT_EQ(Diags.getNumErrors(), 1u);
}

TEST_F(CompilationTest, HandledJobPreservesRedirectionsAndPostCallback) {
  C->Redirect({std::nullopt, "executor-stdout", "executor-stderr"});
  unsigned Calls = 0, Completed = 0;
  C->setJobExecutor([&](const Command &Job, auto Redirects, auto &, bool &) {
    ++Calls;
    EXPECT_EQ(&Job, &*C->getJobs().begin());
    EXPECT_EQ(Redirects.size(), 3u);
    EXPECT_EQ(Redirects[1], "executor-stdout");
    EXPECT_EQ(Redirects[2], "executor-stderr");
    return std::optional<int>(0);
  });
  C->setPostCallback([&](const Command &, int Result) {
    ++Completed;
    EXPECT_EQ(Result, 0);
  });
  const Command *Failed = nullptr;
  EXPECT_EQ(C->ExecuteCommand(*C->getJobs().begin(), Failed), 0);
  EXPECT_EQ(Calls, 1u);
  EXPECT_EQ(Completed, 1u);
  EXPECT_EQ(Failed, nullptr);
  EXPECT_EQ(Diags.getNumErrors(), 0u);
}

TEST_F(CompilationTest, LaunchFailureIsDiagnosed) {
  C->setJobExecutor(
      [](const Command &, auto, std::string &Error, bool &Failed) {
        Error = "linker worker unavailable";
        Failed = true;
        return std::optional<int>(-1);
      });
  const Command *Failed = nullptr;
  EXPECT_EQ(C->ExecuteCommand(*C->getJobs().begin(), Failed), 1);
  EXPECT_EQ(Failed, &*C->getJobs().begin());
  EXPECT_NE(Client.Message.find("linker worker unavailable"),
            std::string::npos);
}

TEST_F(CompilationTest, ToolFailurePreservesDiagnosticsAndCleanup) {
  llvm::SmallString<128> Temporary;
  ASSERT_FALSE(
      llvm::sys::fs::createTemporaryFile("clang-executor", "o", Temporary));
  C->addTempFile(Temporary.c_str());
  C->setJobExecutor([](const Command &, auto, auto &, bool &) {
    return std::optional<int>(42);
  });
  llvm::SmallVector<std::pair<int, const Command *>, 1> Failures;
  D.ExecuteCompilation(*C, Failures);
  ASSERT_EQ(Failures.size(), 1u);
  EXPECT_EQ(Failures[0].first, 42);
  EXPECT_NE(Client.Message.find("linker command failed with exit code 42"),
            std::string::npos);
  C.reset();
  EXPECT_FALSE(llvm::sys::fs::exists(Temporary));
}

TEST_F(CompilationTest, LogOnlyDoesNotExecuteJobs) {
  C->setJobExecutor([](const Command &, auto, auto &, bool &) {
    ADD_FAILURE() << "Log-only compilation must not invoke the executor";
    return std::optional<int>(1);
  });
  const Command *Failed = nullptr;
  EXPECT_EQ(C->ExecuteCommand(*C->getJobs().begin(), Failed, true), 0);
  EXPECT_EQ(Failed, nullptr);
}

TEST_F(CompilationTest, GroupedJobsDoNotBypassExecutor) {
  C.reset(D.BuildCompilation({"clang", "--offload-jobs=2", "-nostdlib",
                              "input.o", "-o", Output.c_str()}));
  ASSERT_TRUE(C);
  ASSERT_FALSE(C->containsError());
  const Tool &Creator = C->getJobs().begin()->getCreator();
  C->getJobs().clear();
  ActionList Inputs;
  for (StringRef Arch : {"first", "second"}) {
    auto *Action = C->MakeAction<LinkJobAction>(Inputs, types::TY_Image);
    C->addCommand(std::make_unique<Command>(
        *Action, Creator, ResponseFileSupport::None(), MissingLinker.c_str(),
        llvm::opt::ArgStringList{}, ArrayRef<InputInfo>{}));
    auto &Job = *C->getJobs().getJobs().back();
    Job.setBoundArch(BoundArch(Arch));
    Job.setOffloadDeviceParallelJobGroup("executor-group");
  }
  unsigned Calls = 0;
  C->setJobExecutor([&](const Command &, auto, auto &, bool &) {
    ++Calls;
    return std::optional<int>(0);
  });
  llvm::SmallVector<std::pair<int, const Command *>, 2> Failures;
  C->ExecuteJobs(C->getJobs(), Failures);
  EXPECT_EQ(Calls, 2u);
  EXPECT_TRUE(Failures.empty());
}
} // namespace
