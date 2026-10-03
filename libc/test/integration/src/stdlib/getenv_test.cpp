//===-- Unittests for getenv ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/getenv.h"

#include "test/IntegrationTest/test.h"

TEST_MAIN([[maybe_unused]] int argc, [[maybe_unused]] char **argv,
          [[maybe_unused]] char **envp) {
  ASSERT_TRUE(LIBC_NAMESPACE::getenv("") == nullptr);
  ASSERT_TRUE(LIBC_NAMESPACE::getenv("=") == nullptr);
  ASSERT_TRUE(LIBC_NAMESPACE::getenv("MISSING ENV VARIABLE") == nullptr);
  ASSERT_FALSE(LIBC_NAMESPACE::getenv("PATH") == nullptr);
  ASSERT_STREQ(LIBC_NAMESPACE::getenv("FRANCE"), "Paris");
  ASSERT_STREQ(LIBC_NAMESPACE::getenv("GERMANY"), "Berlin");
  ASSERT_TRUE(LIBC_NAMESPACE::getenv("FRANC") == nullptr);
  ASSERT_TRUE(LIBC_NAMESPACE::getenv("FRANCE1") == nullptr);

  return 0;
}
