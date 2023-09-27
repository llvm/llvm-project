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

static bool my_streq(const char *lhs, const char *rhs) {
  if (lhs == rhs)
    return true;
  if (((lhs == static_cast<char *>(nullptr)) &&
       (rhs != static_cast<char *>(nullptr))) ||
      ((lhs != static_cast<char *>(nullptr)) &&
       (rhs == static_cast<char *>(nullptr)))) {
    return false;
  }
  const char *l, *r;
  for (l = lhs, r = rhs; *l != '\0' && *r != '\0'; ++l, ++r)
    if (*l != *r)
      return false;

  return *l == '\0' && *r == '\0';
}

static int my_strlen(const char *str) {
  const char *other = str;
  while (*other)
    ++other;
  return static_cast<int>(other - str);
}

TEST_MAIN(int argc, char **argv, char **envp) {
  ASSERT_EQ(argc, 5);
  ASSERT_TRUE(my_streq(argv[1], "%s %c %d"));
  ASSERT_EQ(my_strlen(argv[1]), 8);
  ASSERT_TRUE(my_streq(argv[2], "First arg"));
  ASSERT_EQ(my_strlen(argv[2]), 9);
  ASSERT_TRUE(my_streq(argv[3], "a"));
  ASSERT_EQ(my_strlen(argv[3]), 1);
  ASSERT_TRUE(my_streq(argv[4], "0"));
  ASSERT_EQ(my_strlen(argv[4]), 1);

#ifndef INTEGRATION_DISABLE_PRINTF
  char buf[100];
  ASSERT_EQ(
      LIBC_NAMESPACE::sprintf(buf, argv[1], argv[2], argv[3][0], argv[4][0]),
      14);
  ASSERT_TRUE(my_streq(buf, "First arg a 48"));
#endif

  return 0;
}
