//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for glob and globfree.
///
//===----------------------------------------------------------------------===//

#include "hdr/glob_macros.h"
#include "hdr/types/glob_t.h"
#include "src/glob/glob.h"
#include "src/glob/globfree.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcGlobTest, Dummy) {
  glob_t globbuf = {};
  EXPECT_EQ(LIBC_NAMESPACE::glob("/*", 0, nullptr, &globbuf), GLOB_NOMATCH);

  // globfree should do nothing and not crash.
  LIBC_NAMESPACE::globfree(&globbuf);
}
