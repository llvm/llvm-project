//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file
/// Integration tests for unsetenv.

#include "src/stdlib/getenv.h"
#include "src/stdlib/setenv.h"
#include "src/stdlib/unsetenv.h"
#include "src/string/strcmp.h"
#include "src/unistd/environ.h"

#include "test/IntegrationTest/test.h"

#include <errno.h>

namespace LIBC_NAMESPACE {

TEST_MAIN([[maybe_unused]] int argc, [[maybe_unused]] char **argv,
          [[maybe_unused]] char **envp) {
  // Test: Remove a variable set by setenv
  {
    ASSERT_EQ(setenv("UNSET_VAR", "value", 1), 0);
    ASSERT_TRUE(getenv("UNSET_VAR") != nullptr);

    ASSERT_EQ(unsetenv("UNSET_VAR"), 0);
    ASSERT_TRUE(getenv("UNSET_VAR") == nullptr);
  }

  // Test: Unset non-existent variable succeeds
  {
    ASSERT_EQ(unsetenv("DOES_NOT_EXIST"), 0);
  }

  // Test: Empty name returns EINVAL
  {
    errno = 0;
    ASSERT_EQ(unsetenv(""), -1);
    ASSERT_ERRNO_EQ(EINVAL);
  }

  // Test: Name with '=' returns EINVAL
  {
    errno = 0;
    ASSERT_EQ(unsetenv("BAD=NAME"), -1);
    ASSERT_ERRNO_EQ(EINVAL);
  }

  // Test: Unset then re-set
  {
    ASSERT_EQ(setenv("REUSE_VAR", "first", 1), 0);
    ASSERT_EQ(strcmp(getenv("REUSE_VAR"), "first"), 0);

    ASSERT_EQ(unsetenv("REUSE_VAR"), 0);
    ASSERT_TRUE(getenv("REUSE_VAR") == nullptr);

    ASSERT_EQ(setenv("REUSE_VAR", "second", 1), 0);
    ASSERT_EQ(strcmp(getenv("REUSE_VAR"), "second"), 0);
  }

  // Test: Unset multiple variables
  {
    ASSERT_EQ(setenv("MULTI_A", "a", 1), 0);
    ASSERT_EQ(setenv("MULTI_B", "b", 1), 0);
    ASSERT_EQ(setenv("MULTI_C", "c", 1), 0);

    ASSERT_EQ(unsetenv("MULTI_B"), 0);

    ASSERT_TRUE(getenv("MULTI_A") != nullptr);
    ASSERT_TRUE(getenv("MULTI_B") == nullptr);
    ASSERT_TRUE(getenv("MULTI_C") != nullptr);

    ASSERT_EQ(strcmp(getenv("MULTI_A"), "a"), 0);
    ASSERT_EQ(strcmp(getenv("MULTI_C"), "c"), 0);
  }

  // Test: Unset same variable twice is harmless
  {
    ASSERT_EQ(setenv("DOUBLE_UNSET", "val", 1), 0);
    ASSERT_EQ(unsetenv("DOUBLE_UNSET"), 0);
    ASSERT_EQ(unsetenv("DOUBLE_UNSET"), 0);
    ASSERT_TRUE(getenv("DOUBLE_UNSET") == nullptr);
  }

  // Test: environ is updated and does NOT contain the variable
  {
    ASSERT_EQ(setenv("ENV_CHECK", "val", 1), 0);
    // Verify it is in environ
    bool found = false;
    for (char **env = environ; *env != nullptr; ++env) {
      if (strcmp(*env, "ENV_CHECK=val") == 0) {
        found = true;
        break;
      }
    }
    ASSERT_TRUE(found);

    // Now unset it
    ASSERT_EQ(unsetenv("ENV_CHECK"), 0);

    // Verify it is NOT in environ
    found = false;
    for (char **env = environ; *env != nullptr; ++env) {
      bool match = true;
      const char *prefix = "ENV_CHECK=";
      for (size_t i = 0; i < 10; ++i) {
        if ((*env)[i] == '\0' || (*env)[i] != prefix[i]) {
          match = false;
          break;
        }
      }
      if (match) {
        found = true;
        break;
      }
    }
    ASSERT_FALSE(found);
  }

  return 0;
}

} // namespace LIBC_NAMESPACE
