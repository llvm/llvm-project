//===-- Unittests for getenv ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/IntegrationTest/test.h"

#include <stdlib.h> // getenv

TEST_MAIN(int argc, char **argv, char **envp) {
  ASSERT_STREQ(getenv(""), nullptr);
  ASSERT_STREQ(getenv("="), nullptr);
  ASSERT_STREQ(getenv("MISSING ENV VARIABLE"), nullptr);
  ASSERT_STRNE(getenv("PATH"), nullptr);
  ASSERT_STREQ(getenv("FRANCE"), "Paris");
  ASSERT_STRNE(getenv("FRANCE"), "Berlin");
  ASSERT_STREQ(getenv("GERMANY"), "Berlin");
  ASSERT_STREQ(getenv("FRANC"), nullptr);
  ASSERT_STREQ(getenv("FRANCE1"), nullptr);
  return 0;
}
