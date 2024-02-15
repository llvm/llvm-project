//===-- Loader test to check args to main ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/IntegrationTest/test.h"

TEST_MAIN(int argc, char **argv, char **envp) {
  ASSERT_TRUE(argc == 4);
  ASSERT_STREQ(argv[1], "1");
  ASSERT_STREQ(argv[2], "2");
  ASSERT_STREQ(argv[3], "3");
  ASSERT_STREQ(envp[0], "FRANCE=Paris");
  ASSERT_STREQ(envp[1], "GERMANY=Berlin");
  ASSERT_STREQ(envp[2], nullptr);
  return 0;
}
