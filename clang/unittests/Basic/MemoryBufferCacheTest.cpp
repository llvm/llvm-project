//===- MemoryBufferCacheTest.cpp - MemoryBufferCache tests ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/MemoryBufferCache.h"
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

TEST(MemoryBufferCacheTest, addBuffer) {
  auto B1 = getBuffer(1);
  auto B2 = getBuffer(2);
  auto B3 = getBuffer(3);
  auto *RawB1 = B1.get();
  auto *RawB2 = B2.get();
  auto *RawB3 = B3.get();

  // Add a few buffers.
  MemoryBufferCache Cache;
  EXPECT_EQ(RawB1, &Cache.addBuffer("1", std::move(B1)));
  EXPECT_EQ(RawB2, &Cache.addBuffer("2", std::move(B2)));
  EXPECT_EQ(RawB3, &Cache.addBuffer("3", std::move(B3)));
  EXPECT_EQ(RawB1, Cache.lookupBuffer("1"));
  EXPECT_EQ(RawB2, Cache.lookupBuffer("2"));
  EXPECT_EQ(RawB3, Cache.lookupBuffer("3"));
  EXPECT_FALSE(Cache.isBufferFinal("1"));
  EXPECT_FALSE(Cache.isBufferFinal("2"));
  EXPECT_FALSE(Cache.isBufferFinal("3"));

  // Remove the middle buffer.
  EXPECT_FALSE(Cache.tryToRemoveBuffer("2"));
  EXPECT_EQ(nullptr, Cache.lookupBuffer("2"));
  EXPECT_FALSE(Cache.isBufferFinal("2"));

  // Replace the middle buffer.
  B2 = getBuffer(2);
  RawB2 = B2.get();
  EXPECT_EQ(RawB2, &Cache.addBuffer("2", std::move(B2)));

  // Check that nothing is final.
  EXPECT_FALSE(Cache.isBufferFinal("1"));
  EXPECT_FALSE(Cache.isBufferFinal("2"));
  EXPECT_FALSE(Cache.isBufferFinal("3"));
}

TEST(MemoryBufferCacheTest, finalizeCurrentBuffers) {
  // Add a buffer.
  MemoryBufferCache Cache;
  auto B1 = getBuffer(1);
  auto *RawB1 = B1.get();
  Cache.addBuffer("1", std::move(B1));
  ASSERT_FALSE(Cache.isBufferFinal("1"));

  // Finalize it.
  Cache.finalizeCurrentBuffers();
  EXPECT_TRUE(Cache.isBufferFinal("1"));
  EXPECT_TRUE(Cache.tryToRemoveBuffer("1"));
  EXPECT_EQ(RawB1, Cache.lookupBuffer("1"));
  EXPECT_TRUE(Cache.isBufferFinal("1"));

  // Repeat.
  auto B2 = getBuffer(2);
  auto *RawB2 = B2.get();
  Cache.addBuffer("2", std::move(B2));
  EXPECT_FALSE(Cache.isBufferFinal("2"));

  Cache.finalizeCurrentBuffers();
  EXPECT_TRUE(Cache.isBufferFinal("1"));
  EXPECT_TRUE(Cache.isBufferFinal("2"));
  EXPECT_TRUE(Cache.tryToRemoveBuffer("1"));
  EXPECT_TRUE(Cache.tryToRemoveBuffer("2"));
  EXPECT_EQ(RawB1, Cache.lookupBuffer("1"));
  EXPECT_EQ(RawB2, Cache.lookupBuffer("2"));
  EXPECT_TRUE(Cache.isBufferFinal("1"));
  EXPECT_TRUE(Cache.isBufferFinal("2"));
}

} // namespace
