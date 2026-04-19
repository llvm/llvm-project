//===- HotswapElfTest.cpp - Unit tests for HotSwap ELF layer --------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"
#include "gtest/gtest.h"
#include <cstring>

using namespace COMGR::hotswap;

static constexpr uint32_t TestBranchGfx9 = 0xBF820000u;
static constexpr uint32_t TestBranchGfx12 = 0xBFA00000u;

// -- encodeSBranch ------------------------------------------------------------

TEST(EncodeSBranch, ForwardBranchGfx9) {
  uint8_t Out[MinInstSize] = {};
  ASSERT_TRUE(encodeSBranch(0, 8, Out, TestBranchGfx9));
  uint32_t Encoded;
  std::memcpy(&Encoded, Out, sizeof(Encoded));
  EXPECT_EQ(Encoded, 0xBF820001u);
}

TEST(EncodeSBranch, BackwardBranchGfx9) {
  uint8_t Out[MinInstSize] = {};
  ASSERT_TRUE(encodeSBranch(16, 0, Out, TestBranchGfx9));
  uint32_t Encoded;
  std::memcpy(&Encoded, Out, sizeof(Encoded));
  EXPECT_EQ(Encoded, 0xBF82FFFBu);
}

TEST(EncodeSBranch, ForwardBranchGfx12) {
  uint8_t Out[MinInstSize] = {};
  ASSERT_TRUE(encodeSBranch(0, 8, Out, TestBranchGfx12));
  uint32_t Encoded;
  std::memcpy(&Encoded, Out, sizeof(Encoded));
  EXPECT_EQ(Encoded, 0xBFA00001u);
}

TEST(EncodeSBranch, UnalignedDeltaFails) {
  uint8_t Out[MinInstSize] = {};
  EXPECT_FALSE(encodeSBranch(0, 7, Out, TestBranchGfx9));
}

TEST(EncodeSBranch, OutOfRangeFails) {
  uint8_t Out[MinInstSize] = {};
  EXPECT_FALSE(encodeSBranch(0, 500000, Out, TestBranchGfx9));
}

TEST(EncodeSBranch, ZeroOffsetBranch) {
  uint8_t Out[MinInstSize] = {};
  ASSERT_TRUE(encodeSBranch(0, MinInstSize, Out, TestBranchGfx9));
  uint32_t Encoded;
  std::memcpy(&Encoded, Out, sizeof(Encoded));
  EXPECT_EQ(Encoded, TestBranchGfx9);
}

// -- ElfView::create ----------------------------------------------------------

TEST(ElfView, RejectsTruncatedInput) {
  uint8_t Garbage[] = {0x7f, 'E', 'L', 'F', 0, 0, 0, 0};
  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Garbage, sizeof(Garbage));
  EXPECT_FALSE((bool)ViewOrErr);
  llvm::consumeError(ViewOrErr.takeError());
}

TEST(ElfView, RejectsNonElfInput) {
  uint8_t NotElf[64] = {};
  llvm::Expected<ElfView> ViewOrErr = ElfView::create(NotElf, sizeof(NotElf));
  EXPECT_FALSE((bool)ViewOrErr);
  llvm::consumeError(ViewOrErr.takeError());
}
