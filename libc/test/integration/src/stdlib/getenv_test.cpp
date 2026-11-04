//===-- Unittests for getenv ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/environ_internal.h"
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

  auto &env = LIBC_NAMESPACE::internal::EnvironmentManager::get_instance();
  // Remove inherited variables that could interfere with the lookup tests.
  ASSERT_EQ(env.unset("LIBC_GETENV_TEST"), 0);
  ASSERT_EQ(env.unset("LIBC_GETENV_TEST_LONG"), 0);

  // Exercise entries shorter than the requested name, including entries
  // without '='. Lookup must stop at the terminator of each entry.
  char *const ORIGINAL = *env.begin();
  static char empty_entry[] = "";
  static char short_entry[] = "LIBC_GETENV_TEST";
  static char long_key[] = "LIBC_GETENV_TEST_LONG=value";
  static char empty_value[] = "LIBC_GETENV_TEST=";
  static char value_with_equals[] = "LIBC_GETENV_TEST=first=second";

  *env.begin() = empty_entry;
  ASSERT_TRUE(LIBC_NAMESPACE::getenv("LIBC_GETENV_TEST") == nullptr);
  *env.begin() = short_entry;
  ASSERT_TRUE(LIBC_NAMESPACE::getenv("LIBC_GETENV_TEST") == nullptr);
  ASSERT_TRUE(LIBC_NAMESPACE::getenv("LIBC_GETENV_TEST_LONG") == nullptr);

  *env.begin() = long_key;
  ASSERT_TRUE(LIBC_NAMESPACE::getenv("LIBC_GETENV_TEST") == nullptr);

  *env.begin() = empty_value;
  ASSERT_STREQ(LIBC_NAMESPACE::getenv("LIBC_GETENV_TEST"), "");

  *env.begin() = value_with_equals;
  ASSERT_STREQ(LIBC_NAMESPACE::getenv("LIBC_GETENV_TEST"), "first=second");

  *env.begin() = ORIGINAL;

  return 0;
}
