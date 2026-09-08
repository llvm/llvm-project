//===-- FSTests.cpp - File system related tests -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FS.h"
#include "TestFS.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

TEST(FSTests, PreambleStatusCacheDriveLetter) {
  PreambleFileStatusCache StatCache("C:/proj/main.cpp");
  llvm::vfs::Status S("fake", llvm::sys::fs::UniqueID(1, 2),
                      std::chrono::system_clock::now(), 0, 0, 8,
                      llvm::sys::fs::file_type::regular_file,
                      llvm::sys::fs::all_all);
  llvm::StringMap<std::string> Files;
  auto FS = buildTestFS(Files);
  StatCache.update(*FS, S, "C:/proj/header.h");
  EXPECT_TRUE(StatCache.lookup("c:/proj/header.h"));
  EXPECT_TRUE(StatCache.lookup("c:\\proj\\header.h"));
  EXPECT_FALSE(StatCache.lookup("C:/proj/main.cpp"));
}

#ifdef _WIN32
TEST(FSTests, PreambleStatusCacheDriveRelativePath) {
  auto FS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  ASSERT_TRUE(FS->addFile("C:/one/header.h", 0,
                          llvm::MemoryBuffer::getMemBuffer("one")));
  ASSERT_TRUE(FS->addFile("C:/two/header.h", 0,
                          llvm::MemoryBuffer::getMemBuffer("two longer")));
  ASSERT_FALSE(FS->setCurrentWorkingDirectory("C:/one"));
  PreambleFileStatusCache Cache("C:/one/main.cc");
  auto Original = Cache.getProducingFS(FS)->status("C:header.h");
  ASSERT_TRUE(Original);
  EXPECT_TRUE(Cache.lookup("C:/one/header.h"));
  EXPECT_FALSE(Cache.lookup("C:header.h"));

  ASSERT_FALSE(FS->setCurrentWorkingDirectory("C:/two"));
  auto Cached = Cache.getConsumingFS(FS)->status("C:header.h");
  auto Actual = FS->status("C:header.h");
  ASSERT_TRUE(Cached);
  ASSERT_TRUE(Actual);
  EXPECT_EQ(Cached->getUniqueID(), Actual->getUniqueID());
  EXPECT_EQ(Cached->getSize(), Actual->getSize());
  EXPECT_NE(Cached->getUniqueID(), Original->getUniqueID());
}
#endif

TEST(FSTests, PreambleStatusCache) {
  llvm::StringMap<std::string> Files;
  Files["x"] = "";
  Files["y"] = "";
  Files["main"] = "";
  auto FS = buildTestFS(Files);

  PreambleFileStatusCache StatCache(testPath("main"));
  auto ProduceFS = StatCache.getProducingFS(FS);
  EXPECT_TRUE(ProduceFS->openFileForRead("x"));
  EXPECT_TRUE(ProduceFS->status("y"));
  EXPECT_TRUE(ProduceFS->status("main"));

  EXPECT_TRUE(StatCache.lookup(testPath("x")).has_value());
  EXPECT_TRUE(StatCache.lookup(testPath("y")).has_value());
  // Main file is not cached.
  EXPECT_FALSE(StatCache.lookup(testPath("main")).has_value());

  llvm::vfs::Status S("fake", llvm::sys::fs::UniqueID(123, 456),
                      std::chrono::system_clock::now(), 0, 0, 1024,
                      llvm::sys::fs::file_type::regular_file,
                      llvm::sys::fs::all_all);
  StatCache.update(*FS, S, "real");
  auto ConsumeFS = StatCache.getConsumingFS(FS);
  EXPECT_FALSE(ConsumeFS->status(testPath("fake")));
  auto Cached = ConsumeFS->status(testPath("real"));
  EXPECT_TRUE(Cached);
  EXPECT_EQ(Cached->getName(), testPath("real"));
  EXPECT_EQ(Cached->getUniqueID(), S.getUniqueID());

  // real and temp/../real should hit the same cache entry.
  // However, the Status returned reflects the actual path requested.
  auto CachedDotDot = ConsumeFS->status(testPath("temp/../real"));
  EXPECT_TRUE(CachedDotDot);
  EXPECT_EQ(CachedDotDot->getName(), testPath("temp/../real"));
  EXPECT_EQ(CachedDotDot->getUniqueID(), S.getUniqueID());
}

} // namespace
} // namespace clangd
} // namespace clang
