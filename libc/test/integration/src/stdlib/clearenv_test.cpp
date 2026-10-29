//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Integration tests for the clearenv function.
///
//===----------------------------------------------------------------------===//

#include "src/stdlib/clearenv.h"
#include "src/stdlib/getenv.h"
#include "src/stdlib/putenv.h"
#include "src/stdlib/setenv.h"
#include "src/stdlib/unsetenv.h"
#include "src/unistd/environ.h"

#include "test/IntegrationTest/test.h"

TEST_MAIN([[maybe_unused]] int argc, [[maybe_unused]] char **argv,
          [[maybe_unused]] char **envp) {
  // Test: Clear environment and verify getenv and environ are null
  {
    ASSERT_EQ(LIBC_NAMESPACE::setenv("PRE_CLEAR_VAR", "hello", 1), 0);
    ASSERT_NE(LIBC_NAMESPACE::getenv("PRE_CLEAR_VAR"), nullptr);

    ASSERT_EQ(LIBC_NAMESPACE::clearenv(), 0);
    ASSERT_EQ(LIBC_NAMESPACE::getenv("PRE_CLEAR_VAR"), nullptr);
    ASSERT_EQ(LIBC_NAMESPACE::environ, nullptr);
  }

  // Test: Adding variables after clearenv using setenv
  {
    ASSERT_EQ(LIBC_NAMESPACE::clearenv(), 0);
    ASSERT_EQ(LIBC_NAMESPACE::environ, nullptr);

    ASSERT_EQ(LIBC_NAMESPACE::setenv("POST_CLEAR_VAR", "world", 1), 0);
    ASSERT_NE(LIBC_NAMESPACE::environ, nullptr);
    ASSERT_NE(LIBC_NAMESPACE::getenv("POST_CLEAR_VAR"), nullptr);
    ASSERT_STREQ(LIBC_NAMESPACE::getenv("POST_CLEAR_VAR"), "world");
    ASSERT_STREQ(LIBC_NAMESPACE::environ[0], "POST_CLEAR_VAR=world");
    ASSERT_EQ(LIBC_NAMESPACE::environ[1], nullptr);
  }

  // Test: Adding variables after clearenv using putenv
  {
    ASSERT_EQ(LIBC_NAMESPACE::clearenv(), 0);
    ASSERT_EQ(LIBC_NAMESPACE::environ, nullptr);

    static char put_buf[] = "PUT_AFTER_CLEAR=test";
    ASSERT_EQ(LIBC_NAMESPACE::putenv(put_buf), 0);
    ASSERT_NE(LIBC_NAMESPACE::environ, nullptr);
    ASSERT_NE(LIBC_NAMESPACE::getenv("PUT_AFTER_CLEAR"), nullptr);
    ASSERT_STREQ(LIBC_NAMESPACE::getenv("PUT_AFTER_CLEAR"), "test");
    ASSERT_STREQ(LIBC_NAMESPACE::environ[0], "PUT_AFTER_CLEAR=test");
    ASSERT_EQ(LIBC_NAMESPACE::environ[1], nullptr);
  }

  // Test: Calling clearenv multiple times is idempotent
  {
    ASSERT_EQ(LIBC_NAMESPACE::clearenv(), 0);
    ASSERT_EQ(LIBC_NAMESPACE::environ, nullptr);
    ASSERT_EQ(LIBC_NAMESPACE::clearenv(), 0);
    ASSERT_EQ(LIBC_NAMESPACE::environ, nullptr);
  }

  // Test: Unsetenv after clearenv succeeds
  {
    ASSERT_EQ(LIBC_NAMESPACE::clearenv(), 0);
    ASSERT_EQ(LIBC_NAMESPACE::unsetenv("NONEXISTENT"), 0);
    ASSERT_EQ(LIBC_NAMESPACE::environ, nullptr);
  }

  // Test: Multiple setenv allocations followed by clear
  {
    ASSERT_EQ(LIBC_NAMESPACE::setenv("MULTI_1", "v1", 1), 0);
    ASSERT_EQ(LIBC_NAMESPACE::setenv("MULTI_2", "v2", 1), 0);
    ASSERT_EQ(LIBC_NAMESPACE::setenv("MULTI_3", "v3", 1), 0);
    ASSERT_STREQ(LIBC_NAMESPACE::getenv("MULTI_1"), "v1");
    ASSERT_STREQ(LIBC_NAMESPACE::getenv("MULTI_2"), "v2");
    ASSERT_STREQ(LIBC_NAMESPACE::getenv("MULTI_3"), "v3");

    ASSERT_EQ(LIBC_NAMESPACE::clearenv(), 0);
    ASSERT_EQ(LIBC_NAMESPACE::environ, nullptr);
    ASSERT_EQ(LIBC_NAMESPACE::getenv("MULTI_1"), nullptr);
    ASSERT_EQ(LIBC_NAMESPACE::getenv("MULTI_2"), nullptr);
    ASSERT_EQ(LIBC_NAMESPACE::getenv("MULTI_3"), nullptr);
  }

  // Test: Pointers obtained from getenv remain valid after clearenv
  {
    ASSERT_EQ(LIBC_NAMESPACE::setenv("RETAINED_VAR", "retained_val", 1), 0);
    char *val = LIBC_NAMESPACE::getenv("RETAINED_VAR");
    ASSERT_NE(val, nullptr);
    ASSERT_STREQ(val, "retained_val");

    ASSERT_EQ(LIBC_NAMESPACE::clearenv(), 0);
    ASSERT_EQ(LIBC_NAMESPACE::getenv("RETAINED_VAR"), nullptr);
    ASSERT_EQ(LIBC_NAMESPACE::environ, nullptr);

    // The buffer itself was not erased/deallocated, so existing pointers
    // remain readable without use-after-free.
    ASSERT_STREQ(val, "retained_val");
  }

  return 0;
}
