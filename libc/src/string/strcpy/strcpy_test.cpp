//===----------------------- Unittests for strcpy -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <string>

#include "src/string/strcpy/strcpy.h"
#include "gtest/gtest.h"

TEST(StrCpyTest, EmptyDest) {
  std::string abc = "abc";
  char dest[4];

  char *result = __llvm_libc::strcpy(dest, abc.c_str());
  ASSERT_EQ(dest, result);
  ASSERT_EQ(std::string(dest), abc);
  ASSERT_EQ(std::string(dest).size(), abc.size());
}

TEST(StrCpyTest, OffsetDest) {
  std::string abc = "abc";
  char dest[7];

  dest[0] = 'x';
  dest[1] = 'y';
  dest[2] = 'z';

  char *result = __llvm_libc::strcpy(dest + 3, abc.c_str());
  ASSERT_EQ(dest + 3, result);
  ASSERT_EQ(std::string(dest), std::string("xyz") + abc);
  ASSERT_EQ(std::string(dest).size(), abc.size() + 3);
}
