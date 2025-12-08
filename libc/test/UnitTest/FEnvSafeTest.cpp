//===-- FEnvSafeTest.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "FEnvSafeTest.h"

#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/architectures.h"
#include "test/UnitTest/ErrnoCheckingTest.h"

namespace LIBC_NAMESPACE_DECL {
namespace testing {

void FEnvSafeTest::PreserveFEnv::check() {
  fenv_t after;
  test.get_fenv(after);
  test.expect_fenv_eq(before, after);
}

void FEnvSafeTest::TearDown() {
  if (!should_be_unchanged) {
    restore_fenv();
  }
  // TODO (PR 135320): Remove this override once all FEnvSafeTest instances are
  // updated to validate or ignore errno.
  libc_errno = 0;
  ErrnoCheckingTest::TearDown();
}

void FEnvSafeTest::get_fenv(fenv_t &fenv) {
  ASSERT_EQ(LIBC_NAMESPACE::fputil::get_env(&fenv), 0);
}

void FEnvSafeTest::set_fenv(const fenv_t &fenv) {
  ASSERT_EQ(LIBC_NAMESPACE::fputil::set_env(&fenv), 0);
}

void FEnvSafeTest::expect_fenv_eq(const fenv_t &before_fenv,
                                  const fenv_t &after_fenv) {
#if defined(LIBC_TARGET_ARCH_IS_AARCH64) && !defined(LIBC_COMPILER_IS_MSVC) && \
    defined(__ARM_FP)
  using FPState = LIBC_NAMESPACE::fputil::FEnv::FPState;
  const FPState &before_state = reinterpret_cast<const FPState &>(before_fenv);
  const FPState &after_state = reinterpret_cast<const FPState &>(after_fenv);

  EXPECT_EQ(before_state.ControlWord, after_state.ControlWord);
  EXPECT_EQ(before_state.StatusWord, after_state.StatusWord);

#elif defined(LIBC_TARGET_ARCH_IS_X86)
  using LIBC_NAMESPACE::cpp::inline_copy;
  using LIBC_NAMESPACE::fputil::internal::X87StateDescriptor;
  if constexpr (sizeof(fenv_t) >=
                sizeof(X87StateDescriptor) + sizeof(uint32_t)) {
    const char *before_fenv_ptr = reinterpret_cast<const char *>(&before_fenv);
    const char *after_fenv_ptr = reinterpret_cast<const char *>(&after_fenv);
    X87StateDescriptor before_x87_state, after_x87_state;
    uint32_t before_mxcsr, after_mxcsr;
    inline_copy<sizeof(X87StateDescriptor)>(
        before_fenv_ptr, reinterpret_cast<char *>(&before_x87_state));
    inline_copy<sizeof(X87StateDescriptor)>(
        after_fenv_ptr, reinterpret_cast<char *>(&after_x87_state));
    inline_copy<sizeof(uint32_t)>(before_fenv_ptr + sizeof(X87StateDescriptor),
                                  reinterpret_cast<char *>(&before_mxcsr));
    inline_copy<sizeof(uint32_t)>(after_fenv_ptr + sizeof(X87StateDescriptor),
                                  reinterpret_cast<char *>(&after_mxcsr));

    EXPECT_EQ(before_x87_state.control_word, after_x87_state.control_word);
    EXPECT_EQ(before_x87_state.status_word, after_x87_state.status_word);
    EXPECT_EQ(before_mxcsr, after_mxcsr);

  } else if constexpr (sizeof(fenv_t) == sizeof(X87StateDescriptor)) {
    const X87StateDescriptor &before_state =
        reinterpret_cast<const X87StateDescriptor &>(before_fenv);
    const X87StateDescriptor &after_state =
        reinterpret_cast<const X87StateDescriptor &>(after_fenv);
    EXPECT_EQ(before_state.control_word, after_state.control_word);
    EXPECT_EQ(before_state.status_word, after_state.status_word);

  } else if constexpr (sizeof(fenv_t) == sizeof(uint64_t)) {
    const uint64_t &before_mxcsr =
        reinterpret_cast<const uint64_t &>(before_fenv);
    const uint64_t &after_mxcsr =
        reinterpret_cast<const uint64_t &>(after_fenv);
    EXPECT_EQ(before_mxcsr, after_mxcsr);

  } else if constexpr (sizeof(fenv_t) == sizeof(uint32_t)) {
    const uint32_t &before_mxcsr =
        reinterpret_cast<const uint32_t &>(before_fenv);
    const uint32_t &after_mxcsr =
        reinterpret_cast<const uint32_t &>(after_fenv);
    EXPECT_EQ(before_mxcsr, after_mxcsr);
  }

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

} // namespace testing
} // namespace LIBC_NAMESPACE_DECL
