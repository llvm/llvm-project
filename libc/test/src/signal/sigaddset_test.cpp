//===----------------------- Unittests for sigaddset ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/errno.h"
#include "include/signal.h"
#include "src/errno/llvmlibc_errno.h"
#include "src/signal/sigaddset.h"

#include "utils/UnitTest/Test.h"

// This tests invalid inputs and ensures errno is properly set.
TEST(SignalTest, SigaddsetInvalid) {
  llvmlibc_errno = 0;
  EXPECT_EQ(__llvm_libc::sigaddset(nullptr, SIGSEGV), -1);
  EXPECT_EQ(llvmlibc_errno, EINVAL);

  sigset_t sigset;
  llvmlibc_errno = 0;
  EXPECT_EQ(__llvm_libc::sigaddset(&sigset, -1), -1);
  EXPECT_EQ(llvmlibc_errno, EINVAL);

  // This doesn't use NSIG because __llvm_libc::sigaddset error checking is
  // against sizeof(sigset_t) not NSIG.
  constexpr int bitsInSigsetT = 8 * sizeof(sigset_t);

  llvmlibc_errno = 0;
  EXPECT_EQ(__llvm_libc::sigaddset(&sigset, bitsInSigsetT + 1), -1);
  EXPECT_EQ(llvmlibc_errno, EINVAL);

  llvmlibc_errno = 0;
  EXPECT_EQ(__llvm_libc::sigaddset(&sigset, 0), -1);
  EXPECT_EQ(llvmlibc_errno, EINVAL);

  llvmlibc_errno = 0;
  EXPECT_EQ(__llvm_libc::sigaddset(&sigset, bitsInSigsetT), 0);
  EXPECT_EQ(llvmlibc_errno, 0);
}
