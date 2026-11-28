//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for gethostid.
///
//===----------------------------------------------------------------------===//

#include "src/unistd/gethostid.h"

#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcGethostidTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcGethostidTest, Basic) {
  long id1 = LIBC_NAMESPACE::gethostid();
  EXPECT_EQ(id1, 1L);
  EXPECT_EQ(static_cast<int>(libc_errno), 0);

  long id2 = LIBC_NAMESPACE::gethostid();
  EXPECT_EQ(id1, id2);
}
