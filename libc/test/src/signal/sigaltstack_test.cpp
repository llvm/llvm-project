//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for sigaltstack.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/signal_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/signal/raise.h"
#include "src/signal/sigaction.h"
#include "src/signal/sigaltstack.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

constexpr int LOCAL_VAR_SIZE = 512;
constexpr int ALT_STACK_SIZE = SIGSTKSZ + LOCAL_VAR_SIZE * 2;
static uint8_t alt_stack[ALT_STACK_SIZE];

using LlvmLibcSigaltstackTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

static bool good_stack;
static void handler(int) {
  // Allocate a large stack variable so that it does not get optimized
  // out or mapped to a register.
  uint8_t var[LOCAL_VAR_SIZE];
  for (int i = 0; i < LOCAL_VAR_SIZE; ++i)
    var[i] = static_cast<uint8_t>(i);
  // Verify that array is completely on the alt_stack.
  for (int i = 0; i < LOCAL_VAR_SIZE; ++i) {
    if (!(uintptr_t(var + i) < uintptr_t(alt_stack + ALT_STACK_SIZE) &&
          uintptr_t(alt_stack) <= uintptr_t(var + i))) {
      good_stack = false;
      return;
    }
  }
  good_stack = true;
}

TEST_F(LlvmLibcSigaltstackTest, SigaltstackRunOnAltStack) {
  struct sigaction action;
  ASSERT_THAT(LIBC_NAMESPACE::sigaction(SIGUSR1, nullptr, &action),
              Succeeds(0));
  action.sa_handler = handler;
  // Indicate that the signal should be delivered on an alternate stack.
  action.sa_flags = SA_ONSTACK;
  ASSERT_THAT(LIBC_NAMESPACE::sigaction(SIGUSR1, &action, nullptr),
              Succeeds(0));

  stack_t ss;
  ss.ss_sp = alt_stack;
  ss.ss_size = ALT_STACK_SIZE;
  ss.ss_flags = 0;
  // Setup the alternate stack.
  ASSERT_THAT(LIBC_NAMESPACE::sigaltstack(&ss, nullptr), Succeeds(0));

  good_stack = false;
  LIBC_NAMESPACE::raise(SIGUSR1);
  EXPECT_TRUE(good_stack);
}

// This tests for invalid input.
TEST_F(LlvmLibcSigaltstackTest, SigaltstackInvalidStack) {
  stack_t ss;
  ss.ss_sp = alt_stack;
  ss.ss_size = 0;
  ss.ss_flags = SS_ONSTACK;
  ASSERT_THAT(LIBC_NAMESPACE::sigaltstack(&ss, nullptr), Fails(EINVAL));

  ss.ss_flags = 0;
  ASSERT_THAT(LIBC_NAMESPACE::sigaltstack(&ss, nullptr), Fails(ENOMEM));

  // Sub-minimum size without SS_DISABLE should fail with ENOMEM.
  ss.ss_size = MINSIGSTKSZ - 1;
  ASSERT_THAT(LIBC_NAMESPACE::sigaltstack(&ss, nullptr), Fails(ENOMEM));

  // SS_DISABLE combined with unsupported flags should fail with EINVAL.
  ss.ss_flags = SS_DISABLE | SS_ONSTACK;
  ASSERT_THAT(LIBC_NAMESPACE::sigaltstack(&ss, nullptr), Fails(EINVAL));
}

TEST_F(LlvmLibcSigaltstackTest, SigaltstackDisableStack) {
  // First, set up a valid alternate stack.
  stack_t ss;
  ss.ss_sp = alt_stack;
  ss.ss_size = ALT_STACK_SIZE;
  ss.ss_flags = 0;
  ASSERT_THAT(LIBC_NAMESPACE::sigaltstack(&ss, nullptr), Succeeds(0));

  // Disable the alternate stack with ss_size = 0 and ss_sp = nullptr.
  stack_t disable_ss;
  disable_ss.ss_sp = nullptr;
  disable_ss.ss_size = 0;
  disable_ss.ss_flags = SS_DISABLE;
  stack_t old_ss;
  ASSERT_THAT(LIBC_NAMESPACE::sigaltstack(&disable_ss, &old_ss), Succeeds(0));
  EXPECT_EQ(old_ss.ss_sp, static_cast<void *>(alt_stack));
  EXPECT_EQ(old_ss.ss_size, static_cast<size_t>(ALT_STACK_SIZE));
  EXPECT_EQ(old_ss.ss_flags, 0);

  // Verify that subsequent query reports SS_DISABLE and zeroed stack info.
  stack_t current_ss;
  ASSERT_THAT(LIBC_NAMESPACE::sigaltstack(nullptr, &current_ss), Succeeds(0));
  EXPECT_EQ(current_ss.ss_flags & SS_DISABLE, SS_DISABLE);
  EXPECT_EQ(current_ss.ss_sp, nullptr);
  EXPECT_EQ(current_ss.ss_size, size_t(0));

  // Verify disabling when ss_size is non-zero sub-minimum and ss_sp is non-null
  // (both should be ignored when SS_DISABLE is set).
  disable_ss.ss_sp = alt_stack;
  disable_ss.ss_size = MINSIGSTKSZ - 1;
  disable_ss.ss_flags = SS_DISABLE;
  EXPECT_THAT(LIBC_NAMESPACE::sigaltstack(&disable_ss, nullptr), Succeeds(0));
}
