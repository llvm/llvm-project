//===-- Unittests for getaxuval -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "src/errno/libc_errno.h"
#include "src/sys/auxv/getauxval.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"
#include <src/string/strstr.h>
#include <sys/auxv.h>

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;

TEST(LlvmLibcGetauxvalTest, Basic) {
  EXPECT_THAT(LIBC_NAMESPACE::getauxval(AT_PAGESZ),
              returns(GT(0ul)).with_errno(EQ(0)));
  const char *filename;
  auto getfilename = [&filename]() {
    auto value = LIBC_NAMESPACE::getauxval(AT_EXECFN);
    filename = reinterpret_cast<const char *>(value);
    return value;
  };
  EXPECT_THAT(getfilename(), returns(NE(0ul)).with_errno(EQ(0)));
  ASSERT_TRUE(LIBC_NAMESPACE::strstr(filename, "getauxval_test") != nullptr);
}
