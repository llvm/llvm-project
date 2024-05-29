//===-- FEnvSafeTest.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "FEnvSafeTest.h"

#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/macros/properties/architectures.h"

namespace LIBC_NAMESPACE::testing {

void FEnvSafeTest::PreserveFEnv::check() {
  fenv_t after;
  test.get_fenv(after);
  test.expect_fenv_eq(before, after);
}

void FEnvSafeTest::TearDown() {
  if (!should_be_unchanged) {
    restore_fenv();
  }
}

void FEnvSafeTest::get_fenv(fenv_t &fenv) {
  ASSERT_EQ(LIBC_NAMESPACE::fputil::get_env(&fenv), 0);
}

void FEnvSafeTest::set_fenv(const fenv_t &fenv) {
  ASSERT_EQ(LIBC_NAMESPACE::fputil::set_env(&fenv), 0);
}

void FEnvSafeTest::expect_fenv_eq(const fenv_t &before_fenv,
                                  const fenv_t &after_fenv) {
#if defined(LIBC_TARGET_ARCH_IS_AARCH64)
  using FPState = LIBC_NAMESPACE::fputil::FEnv::FPState;
  const FPState &before_state = reinterpret_cast<const FPState &>(before_fenv);
  const FPState &after_state = reinterpret_cast<const FPState &>(after_fenv);

  EXPECT_EQ(before_state.ControlWord, after_state.ControlWord);
  EXPECT_EQ(before_state.StatusWord, after_state.StatusWord);

#elif defined(LIBC_TARGET_ARCH_IS_X86) && !defined(__APPLE__)
  using LIBC_NAMESPACE::fputil::internal::FPState;
  const FPState &before_state = reinterpret_cast<const FPState &>(before_fenv);
  const FPState &after_state = reinterpret_cast<const FPState &>(after_fenv);

#if defined(_WIN32)
  EXPECT_EQ(before_state.control_word, after_state.control_word);
  EXPECT_EQ(before_state.status_word, after_state.status_word);
#elif defined(__APPLE__)
  EXPECT_EQ(before_state.control_word, after_state.control_word);
  EXPECT_EQ(before_state.status_word, after_state.status_word);
  EXPECT_EQ(before_state.mxcsr, after_state.mxcsr);
#else
  EXPECT_EQ(before_state.x87_status.control_word,
            after_state.x87_status.control_word);
  EXPECT_EQ(before_state.x87_status.status_word,
            after_state.x87_status.status_word);
  EXPECT_EQ(before_state.mxcsr, after_state.mxcsr);
#endif

#elif defined(LIBC_TARGET_ARCH_IS_ARM) && defined(__ARM_FP)
  using LIBC_NAMESPACE::fputil::FEnv;
  const FEnv &before_state = reinterpret_cast<const FEnv &>(before_fenv);
  const FEnv &after_state = reinterpret_cast<const FEnv &>(after_fenv);

  EXPECT_EQ(before_state.fpscr, after_state.fpscr);

#elif defined(LIBC_TARGET_ARCH_IS_ANY_RISCV)
  const uint32_t &before_fcsr = reinterpret_cast<const uint32_t &>(before_fenv);
  const uint32_t &after_fcsr = reinterpret_cast<const uint32_t &>(after_fenv);
  EXPECT_EQ(before_fcsr, after_fcsr);

#else
  // No arch-specific `fenv_t` support, so nothing to compare.

#endif
}

} // namespace LIBC_NAMESPACE::testing
