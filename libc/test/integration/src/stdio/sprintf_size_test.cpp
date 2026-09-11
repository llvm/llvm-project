//===-- Unittests for getenv ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stddef.h>

#ifndef INTEGRATION_DISABLE_PRINTF
#include "src/stdio/sprintf.h"
#endif

#include "test/IntegrationTest/test.h"

static int my_strlen(const char *str) {
  const char *other = str;
  while (*other)
    ++other;
  return static_cast<int>(other - str);
}

TEST_MAIN(int argc, char **argv, [[maybe_unused]] char **envp) {
  ASSERT_EQ(argc, 5);
  ASSERT_STREQ(argv[1], "%s %c %d");
  ASSERT_EQ(my_strlen(argv[1]), 8);
  ASSERT_STREQ(argv[2], "First arg");
  ASSERT_EQ(my_strlen(argv[2]), 9);
  ASSERT_STREQ(argv[3], "a");
  ASSERT_EQ(my_strlen(argv[3]), 1);
  ASSERT_STREQ(argv[4], "0");
  ASSERT_EQ(my_strlen(argv[4]), 1);

#ifndef INTEGRATION_DISABLE_PRINTF
  char buf[100];
  ASSERT_EQ(
      LIBC_NAMESPACE::sprintf(buf, argv[1], argv[2], argv[3][0], argv[4][0]),
      14);
  ASSERT_STREQ(buf, "First arg a 48");
#endif

  return 0;
}
