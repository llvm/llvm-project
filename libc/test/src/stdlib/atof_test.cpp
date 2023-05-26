//===-- Unittests for atof ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/errno/libc_errno.h"
#include "src/stdlib/atof.h"

#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <limits.h>
#include <stddef.h>

using __llvm_libc::testing::ErrnoSetterMatcher::Succeeds;

// This is just a simple test to make sure that this function works at all. It's
// functionally identical to strtod so the bulk of the testing is there.
TEST(LlvmLibcAToFTest, SimpleTest) {
  __llvm_libc::fputil::FPBits<double> expected_fp =
      __llvm_libc::fputil::FPBits<double>(uint64_t(0x405ec00000000000));

  libc_errno = 0;
  EXPECT_THAT(__llvm_libc::atof("123"),
              Succeeds<double>(static_cast<double>(expected_fp)));
}

TEST(LlvmLibcAToFTest, FailedParsingTest) {
  libc_errno = 0;
  // atof does not flag errors.
  EXPECT_THAT(__llvm_libc::atof("???"), Succeeds<double>(0.0));
}
