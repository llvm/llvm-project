//===-- FSTests.cpp - File system related tests -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "FS.h"
#include "TestFS.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

TEST(FSTests, PreambleStatusCache) {
  llvm::StringMap<std::string> Files;
  Files["x"] = "";
  Files["y"] = "";
  Files["main"] = "";
  auto FS = buildTestFS(Files);
  FS->setCurrentWorkingDirectory(testRoot());

  PreambleFileStatusCache StatCache(testPath("main"));
  auto ProduceFS = StatCache.getProducingFS(FS);
  EXPECT_TRUE(ProduceFS->openFileForRead("x"));
  EXPECT_TRUE(ProduceFS->status("y"));
  EXPECT_TRUE(ProduceFS->status("main"));

  EXPECT_TRUE(StatCache.lookup(testPath("x")).hasValue());
  EXPECT_TRUE(StatCache.lookup(testPath("y")).hasValue());
  // Main file is not cached.
  EXPECT_FALSE(StatCache.lookup(testPath("main")).hasValue());

  vfs::Status S("fake", llvm::sys::fs::UniqueID(0, 0),
                std::chrono::system_clock::now(), 0, 0, 1024,
                llvm::sys::fs::file_type::regular_file, llvm::sys::fs::all_all);
  StatCache.update(*FS, S);
  auto ConsumeFS = StatCache.getConsumingFS(FS);
  auto Cached = ConsumeFS->status(testPath("fake"));
  EXPECT_TRUE(Cached);
  EXPECT_EQ(Cached->getName(), S.getName());
}

} // namespace
} // namespace clangd
} // namespace clang
