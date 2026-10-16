//===-- Unittests for perror ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/perror.h"

#include "src/__support/libc_errno.h"
#include "test/UnitTest/Test.h"

// The standard says perror prints directly to stderr and returns nothing. This
// makes it rather difficult to test automatically.

// TODO: figure out redirecting stderr so this test can check correctness.
TEST(LlvmLibcPerrorTest, PrintOut) {
  LIBC_NAMESPACE::libc_errno = 0;
  constexpr char simple[] = "A simple string";
  LIBC_NAMESPACE::perror(simple);

  // stick to stdc errno values, specifically 0, EDOM, ERANGE, and EILSEQ.
  LIBC_NAMESPACE::libc_errno = EDOM;
  LIBC_NAMESPACE::perror("Print this and an error");

  LIBC_NAMESPACE::libc_errno = EILSEQ;
  LIBC_NAMESPACE::perror("\0 shouldn't print this.");

  LIBC_NAMESPACE::libc_errno = ERANGE;
  LIBC_NAMESPACE::perror(nullptr);
}
