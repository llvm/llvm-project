//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Integration tests for unsetenv.
///
//===----------------------------------------------------------------------===//

#include "src/__support/alloc-checker.h"
#include "src/stdlib/environ_internal.h"
#include "src/stdlib/getenv.h"
#include "src/stdlib/setenv.h"
#include "src/stdlib/unsetenv.h"
#include "src/string/memory_utils/inline_memcpy.h"
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

  // Test: Empty name sets errno to EINVAL
  {
    errno = 0;
    ASSERT_EQ(unsetenv(""), -1);
    ASSERT_ERRNO_EQ(EINVAL);
  }

  // Test: Name with '=' sets errno to EINVAL
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

  // Test: Unset removes all duplicate instances of a variable
  {
    ASSERT_EQ(setenv("DUP_VAL1", "1", 1), 0);
    ASSERT_EQ(setenv("DUP_VAL2", "2", 1), 0);

    AllocChecker ac;
    char *d1 = new (ac) char[14];
    ASSERT_TRUE(ac);
    const char *s1 = "DUP_VAR=first";
    inline_memcpy(d1, s1, 14);

    char *d2 = new (ac) char[15];
    ASSERT_TRUE(ac);
    const char *s2 = "DUP_VAR=second";
    inline_memcpy(d2, s2, 15);

    internal::EnvironmentManager &mgr =
        internal::EnvironmentManager::get_instance();
    char **env_array = mgr.begin();
    size_t count = mgr.size();

    size_t idx1 = count;
    size_t idx2 = count;
    for (size_t i = 0; i < count; ++i) {
      cpp::string_view curr(env_array[i]);
      if (curr.starts_with("DUP_VAL1="))
        idx1 = i;
      if (curr.starts_with("DUP_VAL2="))
        idx2 = i;
    }
    ASSERT_TRUE(idx1 < count);
    ASSERT_TRUE(idx2 < count);

    delete[] env_array[idx1];
    delete[] env_array[idx2];

    env_array[idx1] = d1;
    env_array[idx2] = d2;

    ASSERT_EQ(unsetenv("DUP_VAR"), 0);
    ASSERT_TRUE(getenv("DUP_VAR") == nullptr);
  }

  return 0;
}

} // namespace LIBC_NAMESPACE
