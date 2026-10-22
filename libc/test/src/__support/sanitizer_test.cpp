//===-- Unittests for sanitizer utilities ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/macros/sanitizer.h"
#include "test/UnitTest/Test.h"

// Define a fallback in the test binary so ignore_object<true> can link
// across all platforms (including Mach-O and COFF) when LeakSanitizer is
// disabled.
extern "C" [[gnu::weak]] void __lsan_ignore_object(const void *) {}

TEST(LlvmLibcSanitizerTest, LsanIgnoreObject) {
  int val = 42;

  // Verify explicit invocation of the disabled path compiles and executes.
  LIBC_NAMESPACE::sanitizer::ignore_object<false>(&val);
  LIBC_NAMESPACE::sanitizer::ignore_object<false>(nullptr);

  // Verify explicit invocation of the enabled path compiles and executes.
  // When running without LeakSanitizer, the weak __lsan_ignore_object symbol
  // resolves to nullptr, safely bypassing the call.
  LIBC_NAMESPACE::sanitizer::ignore_object<true>(&val);
  LIBC_NAMESPACE::sanitizer::ignore_object<true>(nullptr);

  // Verify the default template argument (HAS_LEAK_SANITIZER).
  LIBC_NAMESPACE::sanitizer::ignore_object(&val);
  LIBC_NAMESPACE::sanitizer::ignore_object(nullptr);

  // Verify the macro wrapper.
  LIBC_LSAN_IGNORE_OBJECT(&val);
  LIBC_LSAN_IGNORE_OBJECT(nullptr);
}
