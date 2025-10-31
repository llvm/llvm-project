//===- unittests/libclang/DependencyScanningCAPITests.cpp ---------------- ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-c/Dependencies.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningService.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace tooling;
using namespace dependencies;

TEST(DependencyScanningCAPITests, DependencyScanningFSCacheOutOfDate) {
  // This test is setup to have two out-of-date file system cache entries,
  // one is negatively stat cached, the other has its size changed.
  //  - `include/b.h` is negatively stat cached.
  //  - `include/a.h` has its size changed.

  CXDependencyScannerServiceOptions ServiceOptions =
      clang_experimental_DependencyScannerServiceOptions_create();
  clang_experimental_DependencyScannerServiceOptions_setCacheNegativeStats(
      ServiceOptions, true);
  CXDependencyScannerService Service =
      clang_experimental_DependencyScannerService_create_v1(ServiceOptions);
  CXDependencyScannerWorker Worker =
      clang_experimental_DependencyScannerWorker_create_v0(Service);

  // Set up the directory structure before scanning.
  // - `/tmp/include/a.h`
  // - `/tmp/include2/b.h`
  llvm::SmallString<128> Dir;
  ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory("tmp", Dir));

  llvm::SmallString<128> Include = Dir;
  llvm::sys::path::append(Include, "include");
  ASSERT_FALSE(llvm::sys::fs::create_directories(Include));

  llvm::SmallString<128> Include2 = Dir;
  llvm::sys::path::append(Include2, "include2");
  ASSERT_FALSE(llvm::sys::fs::create_directories(Include2));

  llvm::SmallString<128> HeaderA = Include;
  llvm::sys::path::append(HeaderA, "a.h");
  {
    std::error_code EC;
    llvm::raw_fd_ostream HeaderAFile(HeaderA, EC);
    ASSERT_FALSE(EC);
  }

  // Initially, we keep include/b.h missing and only create include2/b.h.
  llvm::SmallString<128> HeaderB2 = Include2;
  llvm::sys::path::append(HeaderB2, "b.h");
  {
    std::error_code EC;
    llvm::raw_fd_ostream HeaderB2File(HeaderB2, EC);
    ASSERT_FALSE(EC);
  }

  llvm::SmallString<128> TU = Dir;
  llvm::sys::path::append(TU, "tu.c");
  {
    std::error_code EC;
    llvm::raw_fd_ostream TUFile(TU, EC);
    ASSERT_FALSE(EC);
    TUFile << R"(
      #include "a.h"
      #include "b.h"
    ")";
  }

  const char *Argv[] = {"-c", TU.c_str(),      "-I", Include.c_str(),
                        "-I", Include2.c_str()};
  size_t Argc = std::size(Argv);
  CXDependencyScannerWorkerScanSettings ScanSettings =
      clang_experimental_DependencyScannerWorkerScanSettings_create(
          Argc, Argv, /*ModuleName=*/nullptr, /*WorkingDirectory=*/Dir.c_str(),
          /*MLOContext=*/nullptr, /*MLO=*/nullptr);

  CXDepGraph Graph = nullptr;
  CXErrorCode ScanResult =
      clang_experimental_DependencyScannerWorker_getDepGraph(
          Worker, ScanSettings, &Graph);
  ASSERT_EQ(ScanResult, CXError_Success);

  // Change the size of include/a.h.
  {
    std::error_code EC;
    llvm::raw_fd_ostream HeaderAFile(HeaderA, EC);
    ASSERT_FALSE(EC);
    HeaderAFile << "// New content!\n";
  }

  // Populate include/b.h.
  llvm::SmallString<128> HeaderB = Include;
  llvm::sys::path::append(HeaderB, "b.h");
  {
    std::error_code EC;
    llvm::raw_fd_ostream HeaderBFile(HeaderB, EC);
    ASSERT_FALSE(EC);
  }

  // Directory structure after the change.
  // - `/tmp/include/`
  //     - `a.h` ==> size has changed.
  //     - `b.h` ==> now created.
  // - `/tmp/include2/b.h`

  CXDepScanFSOutOfDateEntrySet Entries =
      clang_experimental_DependencyScannerService_getFSCacheOutOfDateEntrySet(
          Service);

  size_t NumEntries =
      clang_experimental_DepScanFSCacheOutOfDateEntrySet_getNumOfEntries(
          Entries);
  EXPECT_EQ(NumEntries, 2u);

  bool CheckedNegativelyCached = false;
  bool CheckedSizeChanged = false;
  for (size_t Idx = 0; Idx < NumEntries; Idx++) {
    CXDepScanFSOutOfDateEntry Entry =
        clang_experimental_DepScanFSCacheOutOfDateEntrySet_getEntry(Entries,
                                                                    Idx);
    CXDepScanFSCacheOutOfDateKind Kind =
        clang_experimental_DepScanFSCacheOutOfDateEntry_getKind(Entry);
    CXString Path =
        clang_experimental_DepScanFSCacheOutOfDateEntry_getPath(Entry);
    ASSERT_TRUE(Kind == NegativelyCached || Kind == SizeChanged);
    switch (Kind) {
    case NegativelyCached:
      EXPECT_STREQ(clang_getCString(Path), HeaderB.c_str());
      CheckedNegativelyCached = true;
      break;
    case SizeChanged:
      EXPECT_STREQ(clang_getCString(Path), HeaderA.c_str());
      EXPECT_EQ(
          clang_experimental_DepScanFSCacheOutOfDateEntry_getCachedSize(Entry),
          0u);
      EXPECT_EQ(
          clang_experimental_DepScanFSCacheOutOfDateEntry_getActualSize(Entry),
          16u);
      CheckedSizeChanged = true;
      break;
    }
  }

  EXPECT_TRUE(CheckedNegativelyCached && CheckedSizeChanged);

  clang_experimental_DepScanFSCacheOutOfDateEntrySet_disposeSet(Entries);
  clang_experimental_DepGraph_dispose(Graph);
  clang_experimental_DependencyScannerServiceOptions_dispose(ServiceOptions);
  clang_experimental_DependencyScannerWorkerScanSettings_dispose(ScanSettings);
  clang_experimental_DependencyScannerWorker_dispose_v0(Worker);
  clang_experimental_DependencyScannerService_dispose_v0(Service);
}
