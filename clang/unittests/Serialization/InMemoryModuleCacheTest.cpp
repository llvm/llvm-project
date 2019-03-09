//===- InMemoryModuleCacheTest.cpp - InMemoryModuleCache tests ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Serialization/InMemoryModuleCache.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;

namespace {

std::unique_ptr<MemoryBuffer> getBuffer(int I) {
  SmallVector<char, 8> Bytes;
  raw_svector_ostream(Bytes) << "data:" << I;
  return MemoryBuffer::getMemBuffer(StringRef(Bytes.data(), Bytes.size()), "",
                                    /* RequiresNullTerminator = */ false);
}

TEST(InMemoryModuleCacheTest, initialState) {
  InMemoryModuleCache Cache;
  EXPECT_EQ(InMemoryModuleCache::Unknown, Cache.getPCMState("B"));
  EXPECT_FALSE(Cache.isPCMFinal("B"));
  EXPECT_FALSE(Cache.shouldBuildPCM("B"));

#if !defined(NDEBUG) && GTEST_HAS_DEATH_TEST
  EXPECT_DEATH(Cache.tryToDropPCM("B"), "PCM to remove is unknown");
  EXPECT_DEATH(Cache.finalizePCM("B"), "PCM to finalize is unknown");
#endif
}

TEST(InMemoryModuleCacheTest, addPCM) {
  auto B = getBuffer(1);
  auto *RawB = B.get();

  InMemoryModuleCache Cache;
  EXPECT_EQ(RawB, &Cache.addPCM("B", std::move(B)));
  EXPECT_EQ(InMemoryModuleCache::Tentative, Cache.getPCMState("B"));
  EXPECT_EQ(RawB, Cache.lookupPCM("B"));
  EXPECT_FALSE(Cache.isPCMFinal("B"));
  EXPECT_FALSE(Cache.shouldBuildPCM("B"));

#if !defined(NDEBUG) && GTEST_HAS_DEATH_TEST
  EXPECT_DEATH(Cache.addPCM("B", getBuffer(2)), "Already has a PCM");
  EXPECT_DEATH(Cache.addBuiltPCM("B", getBuffer(2)),
               "Trying to override tentative PCM");
#endif
}

TEST(InMemoryModuleCacheTest, addBuiltPCM) {
  auto B = getBuffer(1);
  auto *RawB = B.get();

  InMemoryModuleCache Cache;
  EXPECT_EQ(RawB, &Cache.addBuiltPCM("B", std::move(B)));
  EXPECT_EQ(InMemoryModuleCache::Final, Cache.getPCMState("B"));
  EXPECT_EQ(RawB, Cache.lookupPCM("B"));
  EXPECT_TRUE(Cache.isPCMFinal("B"));
  EXPECT_FALSE(Cache.shouldBuildPCM("B"));

#if !defined(NDEBUG) && GTEST_HAS_DEATH_TEST
  EXPECT_DEATH(Cache.addPCM("B", getBuffer(2)), "Already has a PCM");
  EXPECT_DEATH(Cache.addBuiltPCM("B", getBuffer(2)),
               "Trying to override finalized PCM");
#endif
}

TEST(InMemoryModuleCacheTest, tryToDropPCM) {
  auto B = getBuffer(1);
  auto *RawB = B.get();

  InMemoryModuleCache Cache;
  EXPECT_EQ(InMemoryModuleCache::Unknown, Cache.getPCMState("B"));
  EXPECT_EQ(RawB, &Cache.addPCM("B", std::move(B)));
  EXPECT_FALSE(Cache.tryToDropPCM("B"));
  EXPECT_EQ(nullptr, Cache.lookupPCM("B"));
  EXPECT_EQ(InMemoryModuleCache::ToBuild, Cache.getPCMState("B"));
  EXPECT_FALSE(Cache.isPCMFinal("B"));
  EXPECT_TRUE(Cache.shouldBuildPCM("B"));

#if !defined(NDEBUG) && GTEST_HAS_DEATH_TEST
  EXPECT_DEATH(Cache.addPCM("B", getBuffer(2)), "Already has a PCM");
  EXPECT_DEATH(Cache.tryToDropPCM("B"),
               "PCM to remove is scheduled to be built");
  EXPECT_DEATH(Cache.finalizePCM("B"), "Trying to finalize a dropped PCM");
#endif

  B = getBuffer(2);
  ASSERT_NE(RawB, B.get());
  RawB = B.get();

  // Add a new one.
  EXPECT_EQ(RawB, &Cache.addBuiltPCM("B", std::move(B)));
  EXPECT_TRUE(Cache.isPCMFinal("B"));

  // Can try to drop again, but this should error and do nothing.
  EXPECT_TRUE(Cache.tryToDropPCM("B"));
  EXPECT_EQ(RawB, Cache.lookupPCM("B"));
}

TEST(InMemoryModuleCacheTest, finalizePCM) {
  auto B = getBuffer(1);
  auto *RawB = B.get();

  InMemoryModuleCache Cache;
  EXPECT_EQ(InMemoryModuleCache::Unknown, Cache.getPCMState("B"));
  EXPECT_EQ(RawB, &Cache.addPCM("B", std::move(B)));

  // Call finalize.
  Cache.finalizePCM("B");
  EXPECT_EQ(InMemoryModuleCache::Final, Cache.getPCMState("B"));
  EXPECT_TRUE(Cache.isPCMFinal("B"));
}

} // namespace
