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
#include "src/sys/swap/swapoff.h"
#include "src/sys/swap/swapon.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::any_of;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LlvmLibcSwaponSwapoffTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

// Enabling or disabling swap requires CAP_SYS_ADMIN and modifies global system
// state, so unprivileged test runs can only exercise the failure path. Calling
// the functions at all is still important to verify that the wrappers execute
// and propagate errno without crashing unexpectedly (e.g. under sanitizers).

TEST_F(LlvmLibcSwaponSwapoffTest, SwaponNonExistentPath) {
  auto path = libc_make_test_file_path("swapon_nonexistent.swap");
  ASSERT_THAT(LIBC_NAMESPACE::swapon(path, 0),
              Fails(any_of(ENOENT, EPERM, EACCES, ENOSYS)));
}

TEST_F(LlvmLibcSwaponSwapoffTest, SwapoffNonExistentPath) {
  auto path = libc_make_test_file_path("swapoff_nonexistent.swap");
  ASSERT_THAT(LIBC_NAMESPACE::swapoff(path),
              Fails(any_of(ENOENT, EPERM, EACCES, ENOSYS)));
}
