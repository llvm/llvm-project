//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Integration tests for putenv.
///
//===----------------------------------------------------------------------===//

#include "src/stdlib/getenv.h"
#include "src/stdlib/putenv.h"
#include "src/unistd/environ.h"

#include "test/IntegrationTest/test.h"

#include <errno.h>

static char set_var[] = "PUTENV_TEST=hello";
static char replace_var[] = "PUTENV_TEST=world";
static char empty_value[] = "PUTENV_EMPTY=";
static char special_chars[] = "PUTENV_SPECIAL=!@#$%^&*()";

TEST_MAIN() {
  // Test: Basic set
  {
    ASSERT_EQ(LIBC_NAMESPACE::putenv(set_var), 0);
    char *value = LIBC_NAMESPACE::getenv("PUTENV_TEST");
    ASSERT_TRUE(value != nullptr);
    ASSERT_STREQ(value, "hello");
  }

  // Test: Overwrite existing variable
  {
    ASSERT_EQ(LIBC_NAMESPACE::putenv(replace_var), 0);
    char *value = LIBC_NAMESPACE::getenv("PUTENV_TEST");
    ASSERT_TRUE(value != nullptr);
    ASSERT_STREQ(value, "world");
  }

  // Test: The pointer itself is used (not a copy)
  {
    // After putenv, getenv should return a pointer into the original string.
    char *value = LIBC_NAMESPACE::getenv("PUTENV_TEST");
    ASSERT_TRUE(value == replace_var + 12); // "PUTENV_TEST=" is 12 chars
  }

  // Test: Empty value
  {
    ASSERT_EQ(LIBC_NAMESPACE::putenv(empty_value), 0);
    char *value = LIBC_NAMESPACE::getenv("PUTENV_EMPTY");
    ASSERT_TRUE(value != nullptr);
    ASSERT_STREQ(value, "");
  }

  // Test: Special characters in value
  {
    ASSERT_EQ(LIBC_NAMESPACE::putenv(special_chars), 0);
    char *value = LIBC_NAMESPACE::getenv("PUTENV_SPECIAL");
    ASSERT_TRUE(value != nullptr);
    ASSERT_STREQ(value, "!@#$%^&*()");
  }

  // Test: No '=' removes the variable (POSIX behavior)
  {
    // First set a variable via putenv
    static char var_to_remove[] = "REMOVE_ME=present";
    ASSERT_EQ(LIBC_NAMESPACE::putenv(var_to_remove), 0);
    ASSERT_TRUE(LIBC_NAMESPACE::getenv("REMOVE_ME") != nullptr);

    // Now call putenv without '=' to remove it
    static char remove_cmd[] = "REMOVE_ME";
    ASSERT_EQ(LIBC_NAMESPACE::putenv(remove_cmd), 0);
    ASSERT_TRUE(LIBC_NAMESPACE::getenv("REMOVE_ME") == nullptr);
  }

  // Test: environ is updated and contains the exact pointer
  {
    bool found = false;
    for (char **env = LIBC_NAMESPACE::environ; *env != nullptr; ++env) {
      if (*env == replace_var) {
        found = true;
        break;
      }
    }
    ASSERT_TRUE(found);
  }

  // Test: Pathological inputs (empty names, empty string) are allowed to match
  // glibc.
  {
    // Empty name with value (=value) should succeed and be found in environ.
    static char empty_name[] = "=value";
    ASSERT_EQ(LIBC_NAMESPACE::putenv(empty_name), 0);
    bool found = false;
    for (char **env = LIBC_NAMESPACE::environ; *env != nullptr; ++env) {
      if (*env == empty_name) {
        found = true;
        break;
      }
    }
    ASSERT_TRUE(found);

    // Just "=" should succeed and be found in environ.
    static char just_equals[] = "=";
    ASSERT_EQ(LIBC_NAMESPACE::putenv(just_equals), 0);
    found = false;
    for (char **env = LIBC_NAMESPACE::environ; *env != nullptr; ++env) {
      if (*env == just_equals) {
        found = true;
        break;
      }
    }
    ASSERT_TRUE(found);

    // Empty string unsets the empty name.
    static char empty_string[] = "";
    ASSERT_EQ(LIBC_NAMESPACE::putenv(empty_string), 0);
    // Verify "=value" is no longer in environ (it should have been unset).
    found = false;
    for (char **env = LIBC_NAMESPACE::environ; *env != nullptr; ++env) {
      if (*env == empty_name) {
        found = true;
        break;
      }
    }
    ASSERT_FALSE(found);
  }

  return 0;
}
