//===---- ObjCModuleTest.cpp - clang-tidy ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangTidyOptions.h"
#include "clang/Basic/LLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "gtest/gtest.h"

namespace clang {
namespace tidy {
namespace test {

TEST(ClangTidyOptionsProvider, InMemoryFileSystems) {
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> FileSystem(
      new llvm::vfs::InMemoryFileSystem);

  StringRef BaseClangTidy = R"(
    Checks: -*,clang-diagnostic-*
  )";
  StringRef Sub1ClangTidy = R"(
    Checks: readability-*
    InheritParentConfig: true
  )";
  StringRef Sub2ClangTidy = R"(
    Checks: bugprone-*,misc-*,clang-diagnostic-*
    InheritParentConfig: false
    )";
  FileSystem->addFile("ProjectRoot/.clang-tidy", 0,
                      llvm::MemoryBuffer::getMemBuffer(BaseClangTidy));
  FileSystem->addFile("ProjectRoot/SubDir1/.clang-tidy", 0,
                      llvm::MemoryBuffer::getMemBuffer(Sub1ClangTidy));
  FileSystem->addFile("ProjectRoot/SubDir1/File.cpp", 0,
                      llvm::MemoryBuffer::getMemBuffer(""));
  FileSystem->addFile("ProjectRoot/SubDir1/SubDir2/.clang-tidy", 0,
                      llvm::MemoryBuffer::getMemBuffer(Sub2ClangTidy));
  FileSystem->addFile("ProjectRoot/SubDir1/SubDir2/File.cpp", 0,
                      llvm::MemoryBuffer::getMemBuffer(""));
  FileSystem->addFile("ProjectRoot/SubDir1/SubDir2/SubDir3/File.cpp", 0,
                      llvm::MemoryBuffer::getMemBuffer(""));

  FileOptionsProvider FileOpt({}, {}, {}, FileSystem);

  ClangTidyOptions File1Options =
      FileOpt.getOptions("ProjectRoot/SubDir1/File.cpp");
  ClangTidyOptions File2Options =
      FileOpt.getOptions("ProjectRoot/SubDir1/SubDir2/File.cpp");
  ClangTidyOptions File3Options =
      FileOpt.getOptions("ProjectRoot/SubDir1/SubDir2/SubDir3/File.cpp");

  ASSERT_TRUE(File1Options.Checks.hasValue());
  EXPECT_EQ(*File1Options.Checks, "-*,clang-diagnostic-*,readability-*");
  ASSERT_TRUE(File2Options.Checks.hasValue());
  EXPECT_EQ(*File2Options.Checks, "bugprone-*,misc-*,clang-diagnostic-*");

  // 2 and 3 should use the same config so these should also be the same.
  EXPECT_EQ(File2Options.Checks, File3Options.Checks);
}

} // namespace test
} // namespace tidy
} // namespace clang
