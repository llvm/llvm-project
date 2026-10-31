//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for sync.
///
//===----------------------------------------------------------------------===//

#include "src/unistd/sync.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcSyncTest, SmokeTest) {
  // sync always succeeds. So, we just call it as a smoke test.
  LIBC_NAMESPACE::sync();
}
