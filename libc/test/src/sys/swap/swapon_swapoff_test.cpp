//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for swapon and swapoff.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/sys_swap_macros.h"
#include "src/sys/swap/swapoff.h"
#include "src/sys/swap/swapon.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::any_of;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LlvmLibcSwaponSwapoffTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcSwaponSwapoffTest, SwaponNonExistentPath) {
  auto path = libc_make_test_file_path("swapon_nonexistent.swap");
  ASSERT_THAT(LIBC_NAMESPACE::swapon(path, 0),
              Fails(any_of(ENOENT, EPERM, EINVAL, ENOSYS)));
}

TEST_F(LlvmLibcSwaponSwapoffTest, SwaponFlags) {
  ASSERT_EQ(SWAP_FLAG_PREFER, 0x8000);
  ASSERT_EQ(SWAP_FLAG_PRIO_MASK, 0x7fff);
  ASSERT_EQ(SWAP_FLAG_PRIO_SHIFT, 0);
  ASSERT_EQ(SWAP_FLAG_DISCARD, 0x10000);

  auto path = libc_make_test_file_path("swapon_flags_nonexistent.swap");
  int flags = SWAP_FLAG_PREFER |
              ((1 << SWAP_FLAG_PRIO_SHIFT) & SWAP_FLAG_PRIO_MASK) |
              SWAP_FLAG_DISCARD;
  ASSERT_THAT(LIBC_NAMESPACE::swapon(path, flags),
              Fails(any_of(ENOENT, EPERM, EINVAL, ENOSYS)));
}

TEST_F(LlvmLibcSwaponSwapoffTest, SwapoffNonExistentPath) {
  auto path = libc_make_test_file_path("swapoff_nonexistent.swap");
  ASSERT_THAT(LIBC_NAMESPACE::swapoff(path),
              Fails(any_of(ENOENT, EPERM, EINVAL, ENOSYS)));
}
