//===-- chunk_test.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "chunk.h"

#include "gtest/gtest.h"

#include <stdlib.h>

static constexpr scudo::uptr HeaderSize = scudo::Chunk::getHeaderSize();
static constexpr scudo::u32 Cookie = 0x41424344U;
static constexpr scudo::u32 InvalidCookie = 0x11223344U;

static void initChecksum(void) {
  if (&scudo::computeHardwareCRC32 && scudo::hasHardwareCRC32())
    scudo::HashAlgorithm = scudo::Checksum::HardwareCRC32;
}

TEST(ScudoChunkTest, ChunkBasic) {
  initChecksum();
  const scudo::uptr Size = 0x100U;
  scudo::Chunk::UnpackedHeader Header = {};
  void *Block = malloc(HeaderSize + Size);
  void *P = reinterpret_cast<void *>(reinterpret_cast<scudo::uptr>(Block) +
                                     HeaderSize);
  scudo::Chunk::storeHeader(Cookie, P, &Header);
  memset(P, 'A', Size);
  scudo::Chunk::loadHeader(Cookie, P, &Header);
  EXPECT_TRUE(scudo::Chunk::isValid(Cookie, P, &Header));
  EXPECT_FALSE(scudo::Chunk::isValid(InvalidCookie, P, &Header));
  EXPECT_DEATH(scudo::Chunk::loadHeader(InvalidCookie, P, &Header), "");
  free(Block);
}

TEST(ScudoChunkTest, ChunkCmpXchg) {
  initChecksum();
  const scudo::uptr Size = 0x100U;
  scudo::Chunk::UnpackedHeader OldHeader = {};
  OldHeader.Origin = scudo::Chunk::Origin::Malloc;
  OldHeader.ClassId = 0x42U;
  OldHeader.SizeOrUnusedBytes = Size;
  OldHeader.State = scudo::Chunk::State::Allocated;
  void *Block = malloc(HeaderSize + Size);
  void *P = reinterpret_cast<void *>(reinterpret_cast<scudo::uptr>(Block) +
                                     HeaderSize);
  scudo::Chunk::storeHeader(Cookie, P, &OldHeader);
  memset(P, 'A', Size);
  scudo::Chunk::UnpackedHeader NewHeader = OldHeader;
  NewHeader.State = scudo::Chunk::State::Quarantined;
  scudo::Chunk::compareExchangeHeader(Cookie, P, &NewHeader, &OldHeader);
  NewHeader = {};
  EXPECT_TRUE(scudo::Chunk::isValid(Cookie, P, &NewHeader));
  EXPECT_EQ(NewHeader.State, scudo::Chunk::State::Quarantined);
  EXPECT_FALSE(scudo::Chunk::isValid(InvalidCookie, P, &NewHeader));
  free(Block);
}

TEST(ScudoChunkTest, CorruptHeader) {
  initChecksum();
  const scudo::uptr Size = 0x100U;
  scudo::Chunk::UnpackedHeader Header = {};
  void *Block = malloc(HeaderSize + Size);
  void *P = reinterpret_cast<void *>(reinterpret_cast<scudo::uptr>(Block) +
                                     HeaderSize);
  scudo::Chunk::storeHeader(Cookie, P, &Header);
  memset(P, 'A', Size);
  scudo::Chunk::loadHeader(Cookie, P, &Header);
  // Simulate a couple of corrupted bits per byte of header data.
  for (scudo::uptr I = 0; I < sizeof(scudo::Chunk::PackedHeader); I++) {
    *(reinterpret_cast<scudo::u8 *>(Block) + I) ^= 0x42U;
    EXPECT_DEATH(scudo::Chunk::loadHeader(Cookie, P, &Header), "");
    *(reinterpret_cast<scudo::u8 *>(Block) + I) ^= 0x42U;
  }
  free(Block);
}
