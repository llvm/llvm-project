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
#include "test/UnitTest/Test.h"

TEST(LlvmLibcScandirTest, DummyTest) {
  struct dirent **namelist;
  ASSERT_NE(LIBC_NAMESPACE::scandir(".", &namelist, NULL, NULL), -1);
}
