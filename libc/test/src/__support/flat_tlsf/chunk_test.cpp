//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for flat_tlsf chunk utilities.
///
//===----------------------------------------------------------------------===//

#include "src/__support/flat_tlsf/chunk.h"
#include "src/__support/flat_tlsf/common.h"
#include "src/__support/flat_tlsf/node.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::flat_tlsf::Byte;
using LIBC_NAMESPACE::flat_tlsf::CHUNK_UNIT;
using LIBC_NAMESPACE::flat_tlsf::GAP_BIN_OFFSET;
using LIBC_NAMESPACE::flat_tlsf::GAP_HIGH_SIZE_OFFSET;
using LIBC_NAMESPACE::flat_tlsf::GAP_LOW_SIZE_OFFSET;
using LIBC_NAMESPACE::flat_tlsf::GAP_NODE_OFFSET;
using LIBC_NAMESPACE::flat_tlsf::Node;
namespace chunk = LIBC_NAMESPACE::flat_tlsf::chunk;

TEST(LlvmLibcFlatTlsfChunkTest, GapPointerConversions) {
  alignas(CHUNK_UNIT) Byte bytes[CHUNK_UNIT * 2] = {};
  Byte *base = bytes;
  Byte *end = bytes + CHUNK_UNIT;

  EXPECT_EQ(reinterpret_cast<Byte *>(chunk::gap_base_to_node(base)),
            base + GAP_NODE_OFFSET);
  EXPECT_EQ(reinterpret_cast<Byte *>(chunk::gap_base_to_bin(base)),
            base + GAP_BIN_OFFSET);
  EXPECT_EQ(reinterpret_cast<Byte *>(chunk::gap_base_to_size(base)),
            base + GAP_LOW_SIZE_OFFSET);
  EXPECT_EQ(reinterpret_cast<Byte *>(chunk::gap_end_to_size_and_flag(end)),
            end - GAP_HIGH_SIZE_OFFSET);

  Node *node = chunk::gap_base_to_node(base);
  EXPECT_EQ(chunk::gap_node_to_base(node), base);
  EXPECT_EQ(reinterpret_cast<Byte *>(chunk::gap_node_to_size(node)),
            base + GAP_LOW_SIZE_OFFSET);
  EXPECT_EQ(chunk::end_to_tag(end), end - sizeof(Byte));
}

TEST(LlvmLibcFlatTlsfChunkTest, AlignsByChunkUnit) {
  alignas(CHUNK_UNIT) Byte bytes[CHUNK_UNIT * 3] = {};
  Byte *ptr = bytes + CHUNK_UNIT + 1;

  EXPECT_EQ(chunk::align_down(ptr), bytes + CHUNK_UNIT);
  EXPECT_EQ(chunk::align_up(ptr), bytes + CHUNK_UNIT * 2);
  EXPECT_EQ(chunk::align_up(bytes + CHUNK_UNIT), bytes + CHUNK_UNIT);
}

TEST(LlvmLibcFlatTlsfChunkTest, RequiredChunkSize) {
  // Size + 1 (for tag) rounded up to CHUNK_UNIT
  EXPECT_EQ(chunk::required_chunk_size(0), CHUNK_UNIT);
  EXPECT_EQ(chunk::required_chunk_size(CHUNK_UNIT - 1), CHUNK_UNIT);
  EXPECT_EQ(chunk::required_chunk_size(CHUNK_UNIT), CHUNK_UNIT * 2);
  EXPECT_EQ(chunk::required_chunk_size(CHUNK_UNIT + 5), CHUNK_UNIT * 2);
}

TEST(LlvmLibcFlatTlsfChunkTest, AllocToEnd) {
  Byte *base = reinterpret_cast<Byte *>(uintptr_t{0x1000});
  EXPECT_EQ(chunk::alloc_to_end(base, CHUNK_UNIT), base + CHUNK_UNIT * 2);
}

TEST(LlvmLibcFlatTlsfChunkTest, ReadWriteWord) {
  uint64_t val = 0x0123456789abcdefULL;
  Byte buffer[8] = {};

  chunk::write_word(buffer, val);
  EXPECT_EQ(chunk::read_word<uint64_t>(buffer), val);
}
