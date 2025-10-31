//===- unittest/Tooling/DependencyScanningCASFilesystemTest.cpp -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/DependencyScanning/DependencyScanningCASFilesystem.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningService.h"
#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/CachingOnDiskFileSystem.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace clang::cas;
using namespace clang::cas;
using namespace clang::tooling::dependencies;
using llvm::unittest::TempDir;
using llvm::unittest::TempFile;
using llvm::unittest::TempLink;

TEST(DependencyScanningCASFilesystem, FilenameSpelling) {
  TempDir TestDir("DependencyScanningCASFilesystemTest", /*Unique=*/true);
  TempFile TestFile(TestDir.path("File.h"), "", "#define FOO\n");
#ifndef _WIN32
  TempLink TestLink("File.h", TestDir.path("SymFile.h"));
#else
  // As the create_link uses a hard link on Windows, the full path is
  // required for the link target.
  TempLink TestLink(TestDir.path("File.h"), TestDir.path("SymFile.h"));
#endif

  std::shared_ptr<ObjectStore> CAS = llvm::cas::createInMemoryCAS();
  std::shared_ptr<ActionCache> Cache = llvm::cas::createInMemoryActionCache();
  auto CacheFS = llvm::cantFail(llvm::cas::createCachingOnDiskFileSystem(*CAS));
  DependencyScanningService Service(ScanningMode::DependencyDirectivesScan,
                                    ScanningOutputFormat::Make,
                                    clang::CASOptions(), CAS, Cache, nullptr);
  DependencyScanningCASFilesystem FS(Service, CacheFS);

  EXPECT_EQ(FS.status(TestFile.path()).getError(), std::error_code());
  auto Directives = FS.getDirectiveTokens(TestFile.path());
  ASSERT_TRUE(Directives);
  EXPECT_EQ(Directives->size(), 2u);
  auto DirectivesDots = FS.getDirectiveTokens(TestDir.path("././File.h"));
  ASSERT_TRUE(DirectivesDots);
  EXPECT_EQ(DirectivesDots->size(), 2u);
  auto DirectivesSymlink = FS.getDirectiveTokens(TestLink.path());
  ASSERT_TRUE(DirectivesSymlink);
  EXPECT_EQ(DirectivesSymlink->size(), 2u);
}

TEST(DependencyScanningCASFilesystem, DirectiveScanFailure) {
  TempDir TestDir("DependencyScanningCASFilesystemTest", /*Unique=*/true);
  TempFile TestFile(TestDir.path("python"), "", "import sys\n");

  std::shared_ptr<ObjectStore> CAS = llvm::cas::createInMemoryCAS();
  std::shared_ptr<ActionCache> Cache = llvm::cas::createInMemoryActionCache();
  auto CacheFS = llvm::cantFail(llvm::cas::createCachingOnDiskFileSystem(*CAS));
  DependencyScanningService Service(ScanningMode::DependencyDirectivesScan,
                                    ScanningOutputFormat::Make,
                                    clang::CASOptions(), CAS, Cache, nullptr);
  DependencyScanningCASFilesystem FS(Service, CacheFS);

  EXPECT_EQ(FS.status(TestFile.path()).getError(), std::error_code());
  auto Directives = FS.getDirectiveTokens(TestFile.path());
  ASSERT_FALSE(Directives);

  // Check the cached failure in the action cache.
  {
    DependencyScanningCASFilesystem NewFS(Service, CacheFS);
    EXPECT_EQ(NewFS.status(TestFile.path()).getError(), std::error_code());
    auto Directives = NewFS.getDirectiveTokens(TestFile.path());
    ASSERT_FALSE(Directives);
  }
}
