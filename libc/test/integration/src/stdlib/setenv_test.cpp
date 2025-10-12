//===-- Unittests for setenv ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/getenv.h"
#include "src/stdlib/setenv.h"
#include "src/string/strcmp.h"

#include "test/IntegrationTest/test.h"

#include <errno.h>

TEST_MAIN([[maybe_unused]] int argc, [[maybe_unused]] char **argv,
          [[maybe_unused]] char **envp) {
  // Test: Basic
  {
    // Set a simple environment variable
    ASSERT_EQ(LIBC_NAMESPACE::setenv("SETENV_TEST_VAR", "test_value", 1), 0);

    // Verify it was set
    char *value = LIBC_NAMESPACE::getenv("SETENV_TEST_VAR");
    ASSERT_TRUE(value != nullptr);
    ASSERT_EQ(LIBC_NAMESPACE::strcmp(value, "test_value"), 0);

    // Uncomment after unsetenv is committed
    // LIBC_NAMESPACE::unsetenv("SETENV_TEST_VAR");
  }

  // Test: OverwriteExisting
  {
    // Set initial value
    ASSERT_EQ(LIBC_NAMESPACE::setenv("OVERWRITE_VAR", "original", 1), 0);
    ASSERT_EQ(LIBC_NAMESPACE::strcmp(LIBC_NAMESPACE::getenv("OVERWRITE_VAR"),
                                     "original"),
              0);

    // Overwrite with new value (overwrite = 1)
    ASSERT_EQ(LIBC_NAMESPACE::setenv("OVERWRITE_VAR", "replaced", 1), 0);
    ASSERT_EQ(LIBC_NAMESPACE::strcmp(LIBC_NAMESPACE::getenv("OVERWRITE_VAR"),
                                     "replaced"),
              0);

    // Uncomment after unsetenv is committed
    // LIBC_NAMESPACE::unsetenv("OVERWRITE_VAR");
  }

  // Test: NoOverwriteFlag
  {
    // Set initial value
    ASSERT_EQ(LIBC_NAMESPACE::setenv("NO_OVERWRITE_VAR", "original", 1), 0);
    ASSERT_EQ(LIBC_NAMESPACE::strcmp(LIBC_NAMESPACE::getenv("NO_OVERWRITE_VAR"),
                                     "original"),
              0);

    // Try to set with overwrite = 0 (should not change)
    ASSERT_EQ(LIBC_NAMESPACE::setenv("NO_OVERWRITE_VAR", "ignored", 0), 0);
    ASSERT_EQ(LIBC_NAMESPACE::strcmp(LIBC_NAMESPACE::getenv("NO_OVERWRITE_VAR"),
                                     "original"),
              0);

    // Verify it still works with overwrite = 1
    ASSERT_EQ(LIBC_NAMESPACE::setenv("NO_OVERWRITE_VAR", "changed", 1), 0);
    ASSERT_EQ(LIBC_NAMESPACE::strcmp(LIBC_NAMESPACE::getenv("NO_OVERWRITE_VAR"),
                                     "changed"),
              0);

    // Uncomment after unsetenv is committed
    // LIBC_NAMESPACE::unsetenv("NO_OVERWRITE_VAR");
  }

  // Test: NullName
  {
    errno = 0;
    ASSERT_EQ(LIBC_NAMESPACE::setenv(nullptr, "value", 1), -1);
    ASSERT_ERRNO_EQ(EINVAL);
  }

  // Test: NullValue
  {
    errno = 0;
    ASSERT_EQ(LIBC_NAMESPACE::setenv("NULL_VALUE_VAR", nullptr, 1), -1);
    ASSERT_ERRNO_EQ(EINVAL);
  }

  // Test: EmptyName
  {
    errno = 0;
    ASSERT_EQ(LIBC_NAMESPACE::setenv("", "value", 1), -1);
    ASSERT_ERRNO_EQ(EINVAL);
  }

  // Test: NameWithEquals
  {
    errno = 0;
    ASSERT_EQ(LIBC_NAMESPACE::setenv("BAD=NAME", "value", 1), -1);
    ASSERT_ERRNO_EQ(EINVAL);
  }

  // Test: EmptyValue
  {
    // Empty value is valid - just means variable is set to empty string
    ASSERT_EQ(LIBC_NAMESPACE::setenv("EMPTY_VALUE_VAR", "", 1), 0);

    char *value = LIBC_NAMESPACE::getenv("EMPTY_VALUE_VAR");
    ASSERT_TRUE(value != nullptr);
    ASSERT_EQ(LIBC_NAMESPACE::strcmp(value, ""), 0);

    // Uncomment after unsetenv is committed
    // LIBC_NAMESPACE::unsetenv("EMPTY_VALUE_VAR");
  }

  // Test: MultipleVariables
  {
    // Set multiple different variables
    ASSERT_EQ(LIBC_NAMESPACE::setenv("VAR1", "value1", 1), 0);
    ASSERT_EQ(LIBC_NAMESPACE::setenv("VAR2", "value2", 1), 0);
    ASSERT_EQ(LIBC_NAMESPACE::setenv("VAR3", "value3", 1), 0);

    // Verify all are set correctly
    ASSERT_EQ(LIBC_NAMESPACE::strcmp(LIBC_NAMESPACE::getenv("VAR1"), "value1"),
              0);
    ASSERT_EQ(LIBC_NAMESPACE::strcmp(LIBC_NAMESPACE::getenv("VAR2"), "value2"),
              0);
    ASSERT_EQ(LIBC_NAMESPACE::strcmp(LIBC_NAMESPACE::getenv("VAR3"), "value3"),
              0);

    // Uncomment after unsetenv is committed
    // LIBC_NAMESPACE::unsetenv("VAR1");
    // LIBC_NAMESPACE::unsetenv("VAR2");
    // LIBC_NAMESPACE::unsetenv("VAR3");
  }

  // Test: LongValues
  {
    // Test with longer strings
    const char *long_name = "LONG_VAR_NAME_FOR_TESTING";
    const char *long_value = "This is a fairly long value string to test that "
                             "setenv handles longer strings correctly without "
                             "any memory issues or truncation problems";

    ASSERT_EQ(LIBC_NAMESPACE::setenv(long_name, long_value, 1), 0);
    ASSERT_EQ(
        LIBC_NAMESPACE::strcmp(LIBC_NAMESPACE::getenv(long_name), long_value),
        0);

    // Uncomment after unsetenv is committed
    // LIBC_NAMESPACE::unsetenv(long_name);
  }

  // Test: SpecialCharacters
  {
    // Test with special characters in value (but not in name)
    ASSERT_EQ(LIBC_NAMESPACE::setenv("SPECIAL_CHARS", "!@#$%^&*()", 1), 0);
    ASSERT_EQ(LIBC_NAMESPACE::strcmp(LIBC_NAMESPACE::getenv("SPECIAL_CHARS"),
                                     "!@#$%^&*()"),
              0);

    // Uncomment after unsetenv is committed
    // LIBC_NAMESPACE::unsetenv("SPECIAL_CHARS");
  }

  // Test: ReplaceMultipleTimes
  {
    // Replace the same variable multiple times
    ASSERT_EQ(LIBC_NAMESPACE::setenv("MULTI_REPLACE", "value1", 1), 0);
    ASSERT_EQ(LIBC_NAMESPACE::strcmp(LIBC_NAMESPACE::getenv("MULTI_REPLACE"),
                                     "value1"),
              0);

    ASSERT_EQ(LIBC_NAMESPACE::setenv("MULTI_REPLACE", "value2", 1), 0);
    ASSERT_EQ(LIBC_NAMESPACE::strcmp(LIBC_NAMESPACE::getenv("MULTI_REPLACE"),
                                     "value2"),
              0);

    ASSERT_EQ(LIBC_NAMESPACE::setenv("MULTI_REPLACE", "value3", 1), 0);
    ASSERT_EQ(LIBC_NAMESPACE::strcmp(LIBC_NAMESPACE::getenv("MULTI_REPLACE"),
                                     "value3"),
              0);

    // Uncomment after unsetenv is committed
    // LIBC_NAMESPACE::unsetenv("MULTI_REPLACE");
  }

  return 0;
}
