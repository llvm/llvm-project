//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for the static TLS layout
///
//===----------------------------------------------------------------------===//

#include "src/__support/elf/tls_layout.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::elf::round_up;
using LIBC_NAMESPACE::elf::TLS_TCB_SIZE;
using LIBC_NAMESPACE::elf::TLS_VARIANT_2;
using LIBC_NAMESPACE::elf::TlsLayout;

TEST(LlvmLibcElfTlsLayoutTest, RoundUp) {
  EXPECT_EQ(round_up(0, 8), size_t(0));
  EXPECT_EQ(round_up(1, 8), size_t(8));
  EXPECT_EQ(round_up(8, 8), size_t(8));
  EXPECT_EQ(round_up(9, 8), size_t(16));
  // An alignment of zero or one must leave the value alone.
  EXPECT_EQ(round_up(7, 1), size_t(7));
  EXPECT_EQ(round_up(7, 0), size_t(7));
}

TEST(LlvmLibcElfTlsLayoutTest, EmptyLayout) {
  TlsLayout layout;
  EXPECT_EQ(layout.module_count(), size_t(0));
  EXPECT_EQ(layout.alignment(), TLS_TCB_SIZE);
  if (TLS_VARIANT_2)
    EXPECT_EQ(layout.size(), TLS_TCB_SIZE);
  else
    EXPECT_EQ(layout.size(), size_t(0));
}

TEST(LlvmLibcElfTlsLayoutTest, OffsetsAreOnTheRightSide) {
  TlsLayout layout;
  intptr_t first = layout.add(32, 8);
  intptr_t second = layout.add(16, 8);
  if (TLS_VARIANT_2) {
    // Below the thread pointer, and each module further down than the last.
    EXPECT_LT(first, intptr_t(0));
    EXPECT_LT(second, first);
  } else {
    // Above the thread pointer, past the thread control block.
    EXPECT_GE(first, intptr_t(TLS_TCB_SIZE));
    EXPECT_GT(second, first);
  }
  EXPECT_EQ(layout.module_count(), size_t(2));
}

TEST(LlvmLibcElfTlsLayoutTest, BlocksDoNotOverlap) {
  // Three modules with awkward sizes and alignments. Whatever the variant,
  // each block must fit entirely inside the total and not touch its
  // neighbours.
  struct Mod {
    size_t size, align;
  };
  const Mod mods[] = {{40, 8}, {1, 1}, {96, 64}};

  TlsLayout layout;
  intptr_t offsets[3];
  for (int i = 0; i < 3; ++i)
    offsets[i] = layout.add(mods[i].size, mods[i].align);

  for (int i = 0; i < 3; ++i) {
    // Each block must respect its own alignment requirement.
    size_t magnitude =
        static_cast<size_t>(offsets[i] < 0 ? -offsets[i] : offsets[i]);
    EXPECT_EQ(magnitude % mods[i].align, size_t(0));
    // And must lie inside the block that gets allocated.
    EXPECT_LE(magnitude, layout.size());
  }

  for (int i = 0; i + 1 < 3; ++i) {
    size_t a = static_cast<size_t>(offsets[i] < 0 ? -offsets[i] : offsets[i]);
    size_t b = static_cast<size_t>(offsets[i + 1] < 0 ? -offsets[i + 1]
                                                      : offsets[i + 1]);
    if (TLS_VARIANT_2)
      // b is further from the thread pointer and must clear a's whole block.
      EXPECT_GE(b, a + mods[i + 1].size);
    else
      EXPECT_GE(b, a + mods[i].size);
  }
}

TEST(LlvmLibcElfTlsLayoutTest, AlignmentIsTheLargestSeen) {
  TlsLayout layout;
  layout.add(8, 8);
  layout.add(8, 64);
  layout.add(8, 16);
  EXPECT_EQ(layout.alignment(), size_t(64));
}

TEST(LlvmLibcElfTlsLayoutTest, SizeCoversEveryModule) {
  // The case that motivated this: a small executable followed by a libc.so
  // with a large TLS segment. The total has to cover both.
  TlsLayout layout;
  layout.add(16, 8);
  layout.add(52904, 64);
  EXPECT_GE(layout.size(), size_t(16 + 52904));
  EXPECT_EQ(layout.module_count(), size_t(2));
}

TEST(LlvmLibcElfTlsLayoutTest, ZeroAlignmentIsTreatedAsOne) {
  TlsLayout layout;
  intptr_t offset = layout.add(24, 0);
  EXPECT_EQ(layout.module_count(), size_t(1));
  size_t magnitude = static_cast<size_t>(offset < 0 ? -offset : offset);
  EXPECT_GE(layout.size(), magnitude);
}
