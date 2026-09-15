//===-- Tests for Darwin sigprocmask --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/signal_macros.h"
#include "hdr/types/sigset_t.h"
#include "src/__support/OSUtil/darwin/syscall_wrappers/sigprocmask.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::darwin_syscalls::sigprocmask;

class LlvmLibcDarwinSigprocmaskTest
    : public LIBC_NAMESPACE::testing::ErrnoCheckingTest {
  sigset_t original;

public:
  void SetUp() override {
    ErrnoCheckingTest::SetUp();
    ASSERT_TRUE(sigprocmask(0, nullptr, &original).has_value());
  }

  void TearDown() override {
    ASSERT_TRUE(sigprocmask(SIG_SETMASK, &original, nullptr).has_value());
    ErrnoCheckingTest::TearDown();
  }
};

TEST_F(LlvmLibcDarwinSigprocmaskTest, QueryIgnoresHow) {
  sigset_t empty{};
  ASSERT_TRUE(sigprocmask(SIG_SETMASK, &empty, nullptr).has_value());
  sigset_t observed;
  observed.__signals[0] = ~0UL;
  auto result = sigprocmask(-4, nullptr, &observed);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), 0);
  EXPECT_EQ(observed.__signals[0], 0UL);
  ASSERT_TRUE(sigprocmask(-4, nullptr, nullptr).has_value());
}

TEST_F(LlvmLibcDarwinSigprocmaskTest, BlockAndUnblock) {
  sigset_t empty{};
  ASSERT_TRUE(sigprocmask(SIG_SETMASK, &empty, nullptr).has_value());
  sigset_t mask{};
  mask.__signals[0] = (1UL << (SIGUSR1 - 1)) | (1UL << (SIGUSR2 - 1));
  sigset_t previous;
  previous.__signals[0] = ~0UL;
  ASSERT_TRUE(sigprocmask(SIG_BLOCK, &mask, &previous).has_value());
  EXPECT_EQ(previous.__signals[0], 0UL);
  sigset_t observed;
  observed.__signals[0] = ~0UL;
  ASSERT_TRUE(sigprocmask(0, nullptr, &observed).has_value());
  EXPECT_EQ(observed.__signals[0], mask.__signals[0]);
  ASSERT_TRUE(sigprocmask(SIG_UNBLOCK, &mask, &previous).has_value());
  EXPECT_EQ(previous.__signals[0], mask.__signals[0]);
  ASSERT_TRUE(sigprocmask(0, nullptr, &observed).has_value());
  EXPECT_EQ(observed.__signals[0], 0UL);
}

TEST_F(LlvmLibcDarwinSigprocmaskTest, FullMask) {
  sigset_t mask;
  mask.__signals[0] = ~0UL;
  ASSERT_TRUE(sigprocmask(SIG_SETMASK, &mask, nullptr).has_value());
  sigset_t observed;
  observed.__signals[0] = ~0UL;
  ASSERT_TRUE(sigprocmask(0, nullptr, &observed).has_value());
  EXPECT_EQ(observed.__signals[0],
            0x7FFFFFFFUL & ~((1UL << (SIGKILL - 1)) | (1UL << (SIGSTOP - 1))));
}

TEST_F(LlvmLibcDarwinSigprocmaskTest, InvalidHowPreservesOutputAndErrno) {
  sigset_t empty{};
  ASSERT_TRUE(sigprocmask(SIG_SETMASK, &empty, nullptr).has_value());
  sigset_t mask{};
  mask.__signals[0] = 1UL << (SIGUSR1 - 1);
  sigset_t observed;
  observed.__signals[0] = ~0UL;
  libc_errno = ERANGE;
  auto result = sigprocmask(17, &mask, &observed);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), EINVAL);
  EXPECT_EQ(observed.__signals[0], ~0UL);
  EXPECT_EQ(static_cast<int>(libc_errno), ERANGE);
  ASSERT_TRUE(sigprocmask(0, nullptr, &observed).has_value());
  EXPECT_EQ(observed.__signals[0], 0UL);
  EXPECT_EQ(static_cast<int>(libc_errno), ERANGE);
  libc_errno = 0;
}
