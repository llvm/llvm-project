//===-- Unittests for sprintf ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef INTEGRATION_DISABLE_PRINTF
#include <stdio.h> // sprintf
#endif

#include "test/IntegrationTest/test.h"

TEST_MAIN(int argc, char **argv, char **envp) {
  ASSERT_EQ(argc, 5);
  ASSERT_STREQ(argv[1], "%s %c %d");
  ASSERT_STREQ(argv[2], "First arg");
  ASSERT_STREQ(argv[3], "a");
  ASSERT_STREQ(argv[4], "0");

#ifndef INTEGRATION_DISABLE_PRINTF
  char buf[100];
  ASSERT_EQ(sprintf(buf, argv[1], argv[2], argv[3][0], argv[4][0]), 14);
  ASSERT_STREQ(buf, "First arg a 48");
#endif

  return 0;
}
