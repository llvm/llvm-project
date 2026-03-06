//===-- Unittests for nl_types --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/llvm-libc-macros/nl-types-macros.h"
#include "include/llvm-libc-types/nl_catd.h"
#include "src/nl_types/catclose.h"
#include "src/nl_types/catgets.h"
#include "src/nl_types/catopen.h"
#include "test/UnitTest/ErrnoCheckingTest.h"

using LlvmLibcNlTypesTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcNlTypesTest, CatopenFails) {
  ASSERT_EQ(LIBC_NAMESPACE::catopen("/somepath", NL_CAT_LOCALE),
            reinterpret_cast<nl_catd>(-1));
  ASSERT_ERRNO_EQ(EINVAL);
}

TEST_F(LlvmLibcNlTypesTest, CatcloseFails) {
  ASSERT_EQ(LIBC_NAMESPACE::catclose(nullptr), -1);
}

TEST_F(LlvmLibcNlTypesTest, CatgetsFails) {
  const char *message = "message";
  // Note that we test for pointer equality here, since catgets
  // is expected to return the input argument as-is.
  ASSERT_EQ(LIBC_NAMESPACE::catgets(nullptr, NL_SETD, 1, message),
            const_cast<char *>(message));
}
