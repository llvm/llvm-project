//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for scandir.
///
//===----------------------------------------------------------------------===//

#include "hdr/types/struct_dirent.h"
#include "src/dirent/scandir.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcScandirTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcScandirTest, TestDummy) {
  struct dirent **namelist;
  ASSERT_THAT(LIBC_NAMESPACE::scandir("/tmp", &namelist, NULL, NULL), Succeeds());
}

TEST_F(LlvmLibcScandirTest, TestBadDirname) {
  struct dirent **namelist;
  ASSERT_THAT(LIBC_NAMESPACE::scandir("", &namelist, NULL, NULL), Fails(ENOENT, -1));
}
