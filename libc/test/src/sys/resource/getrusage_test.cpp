//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for getrusage.
///
//===----------------------------------------------------------------------===//

#include "hdr/types/struct_rusage.h"
#include "src/sys/resource/getrusage.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcGetrusageTest, DummyTest) {
  struct rusage usage = {};
  int res = LIBC_NAMESPACE::getrusage(0, &usage);
  EXPECT_NE(res, -1);
}
