//===---------------------- Unittests for strcat --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <string>

#include "src/string/strcat/strcat.h"
#include "gtest/gtest.h"

TEST(StrCatTest, EmptyDest) {
  std::string abc = "abc";
  char dest[4];

  dest[0] = '\0';

  char *result = __llvm_libc::strcat(dest, abc.c_str());
  ASSERT_EQ(dest, result);
  ASSERT_EQ(std::string(dest), abc);
  ASSERT_EQ(std::string(dest).size(), abc.size());
}

TEST(StrCatTest, NonEmptyDest) {
  std::string abc = "abc";
  char dest[7];

  dest[0] = 'x';
  dest[1] = 'y';
  dest[2] = 'z';
  dest[3] = '\0';

  char *result = __llvm_libc::strcat(dest, abc.c_str());
  ASSERT_EQ(dest, result);
  ASSERT_EQ(std::string(dest), std::string("xyz") + abc);
  ASSERT_EQ(std::string(dest).size(), abc.size() + 3);
}
