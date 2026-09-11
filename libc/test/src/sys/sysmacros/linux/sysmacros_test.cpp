//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for sysmacros (major, minor, makedev).
///
//===----------------------------------------------------------------------===//

#include "hdr/types/dev_t.h"
#include "src/sys/sysmacros/major.h"
#include "src/sys/sysmacros/makedev.h"
#include "src/sys/sysmacros/minor.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcSysmacrosTest, Zero) {
  dev_t dev = LIBC_NAMESPACE::makedev(0, 0);
  EXPECT_EQ(dev, static_cast<dev_t>(0));
  EXPECT_EQ(LIBC_NAMESPACE::major(dev), 0u);
  EXPECT_EQ(LIBC_NAMESPACE::minor(dev), 0u);
}

TEST(LlvmLibcSysmacrosTest, StandardLinuxDevices) {
  // /dev/null: major 1, minor 3 -> dev = (1 << 8) | 3 = 0x103
  dev_t dev_null = LIBC_NAMESPACE::makedev(1, 3);
  EXPECT_EQ(dev_null, static_cast<dev_t>(0x103));
  EXPECT_EQ(LIBC_NAMESPACE::major(dev_null), 1u);
  EXPECT_EQ(LIBC_NAMESPACE::minor(dev_null), 3u);

  // /dev/pts/0: major 136 (0x88), minor 0 -> dev = 0x8800
  dev_t dev_pts0 = LIBC_NAMESPACE::makedev(136, 0);
  EXPECT_EQ(dev_pts0, static_cast<dev_t>(0x8800));
  EXPECT_EQ(LIBC_NAMESPACE::major(dev_pts0), 136u);
  EXPECT_EQ(LIBC_NAMESPACE::minor(dev_pts0), 0u);

  // /dev/sda1: major 8, minor 1 -> dev = 0x801
  dev_t dev_sda1 = LIBC_NAMESPACE::makedev(8, 1);
  EXPECT_EQ(dev_sda1, static_cast<dev_t>(0x801));
  EXPECT_EQ(LIBC_NAMESPACE::major(dev_sda1), 8u);
  EXPECT_EQ(LIBC_NAMESPACE::minor(dev_sda1), 1u);
}

TEST(LlvmLibcSysmacrosTest, LargeMinorOrMajor) {
  // Minor >= 256 uses the upper bits of dev_t (bits 20..43).
  // minor = 256 (0x100): lower 8 bits = 0x00, upper bits shifted by 12 =
  // 0x100000.
  dev_t dev1 = LIBC_NAMESPACE::makedev(0, 0x100);
  EXPECT_EQ(dev1, static_cast<dev_t>(0x100000));
  EXPECT_EQ(LIBC_NAMESPACE::major(dev1), 0u);
  EXPECT_EQ(LIBC_NAMESPACE::minor(dev1), 0x100u);

  // Major >= 4096 uses the upper bits of dev_t (bits 44..63).
  // major = 4096 (0x1000): lower 12 bits = 0, upper bits shifted by 32 =
  // 0x100000000000.
  dev_t dev2 = LIBC_NAMESPACE::makedev(0x1000, 0);
  EXPECT_EQ(dev2, static_cast<dev_t>(0x100000000000ULL));
  EXPECT_EQ(LIBC_NAMESPACE::major(dev2), 0x1000u);
  EXPECT_EQ(LIBC_NAMESPACE::minor(dev2), 0u);
}

TEST(LlvmLibcSysmacrosTest, MaxValues) {
  // All bits set
  dev_t dev_max = LIBC_NAMESPACE::makedev(0xffffffffu, 0xffffffffu);
  EXPECT_EQ(dev_max, static_cast<dev_t>(0xffffffffffffffffULL));
  EXPECT_EQ(LIBC_NAMESPACE::major(dev_max), 0xffffffffu);
  EXPECT_EQ(LIBC_NAMESPACE::minor(dev_max), 0xffffffffu);
}

TEST(LlvmLibcSysmacrosTest, RoundTrip) {
  struct TestCase {
    unsigned int maj;
    unsigned int min;
  } test_cases[] = {
      {0x0, 0x0},
      {0x1, 0x1},
      {0xfff, 0xffff},
      {0xffff, 0xfff},
      {0xdeadbeef, 0x12345678},
      {0x87654321, 0xabcdef01},
  };

  for (const auto &tc : test_cases) {
    dev_t d = LIBC_NAMESPACE::makedev(tc.maj, tc.min);
    EXPECT_EQ(LIBC_NAMESPACE::major(d), tc.maj);
    EXPECT_EQ(LIBC_NAMESPACE::minor(d), tc.min);
  }
}
