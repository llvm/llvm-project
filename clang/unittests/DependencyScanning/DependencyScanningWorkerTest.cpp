//===- DependencyScanningWorkerTest.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/DependencyScanning/DependencyScanningWorker.h"
#include "clang/DependencyScanning/DependencyScanningUtils.h"
#include "llvm/Support/FormatVariadic.h"
#include "gtest/gtest.h"
#include <string>

using namespace clang;
using namespace dependencies;

TEST(DependencyScanner, ScanDepsWithDiagConsumer) {
  StringRef CWD = "/root";

  auto VFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  VFS->setCurrentWorkingDirectory(CWD);
  auto Sept = llvm::sys::path::get_separator();
  std::string HeaderPath =
      std::string(llvm::formatv("{0}root{0}header.h", Sept));
  std::string TestPath = std::string(llvm::formatv("{0}root{0}test.cpp", Sept));
  std::string AsmPath = std::string(llvm::formatv("{0}root{0}test.s", Sept));

  VFS->addFile(HeaderPath, 0, llvm::MemoryBuffer::getMemBuffer("\n"));
  VFS->addFile(TestPath, 0,
               llvm::MemoryBuffer::getMemBuffer("#include \"header.h\"\n"));
  VFS->addFile(AsmPath, 0, llvm::MemoryBuffer::getMemBuffer(""));

  DependencyScanningService Service(ScanningMode::DependencyDirectivesScan,
                                    ScanningOutputFormat::Make);
  DependencyScanningWorker Worker(Service, VFS);

  llvm::DenseSet<ModuleID> AlreadySeen;
  FullDependencyConsumer DC(AlreadySeen);
  CallbackActionController AC(nullptr);

  struct EnsureFinishedConsumer : public DiagnosticConsumer {
    bool Finished = false;
    void finish() override { Finished = true; }
  };

  {
    // Check that a successful scan calls DiagConsumer.finish().
    std::vector<std::string> Args = {"clang",
                                     "-target",
                                     "x86_64-apple-macosx10.7",
                                     "-c",
                                     "test.cpp",
                                     "-o"
                                     "test.cpp.o"};

    EnsureFinishedConsumer DiagConsumer;
    bool Success = Worker.computeDependencies(CWD, Args, DC, AC, DiagConsumer);

    EXPECT_TRUE(Success);
    EXPECT_EQ(DiagConsumer.getNumErrors(), 0u);
    EXPECT_TRUE(DiagConsumer.Finished);
  }

  {
    // Check that an invalid command-line, which never enters the scanning
    // action calls DiagConsumer.finish().
    std::vector<std::string> Args = {"clang", "-invalid-arg"};
    EnsureFinishedConsumer DiagConsumer;
    bool Success = Worker.computeDependencies(CWD, Args, DC, AC, DiagConsumer);

    EXPECT_FALSE(Success);
    EXPECT_GE(DiagConsumer.getNumErrors(), 1u);
    EXPECT_TRUE(DiagConsumer.Finished);
  }

  {
    // Check that a valid command line that produces no scanning jobs calls
    // DiagConsumer.finish().
    std::vector<std::string> Args = {"clang",
                                     "-target",
                                     "x86_64-apple-macosx10.7",
                                     "-c",
                                     "-x",
                                     "assembler",
                                     "test.s",
                                     "-o"
                                     "test.cpp.o"};

    EnsureFinishedConsumer DiagConsumer;
    bool Success = Worker.computeDependencies(CWD, Args, DC, AC, DiagConsumer);

    EXPECT_FALSE(Success);
    EXPECT_EQ(DiagConsumer.getNumErrors(), 1u);
    EXPECT_TRUE(DiagConsumer.Finished);
  }
}
