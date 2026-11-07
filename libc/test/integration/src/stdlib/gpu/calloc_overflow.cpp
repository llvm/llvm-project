//===-- GPU calloc overflow test ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/calloc.h"
#include "test/IntegrationTest/test.h"

TEST_MAIN(int, char **, char **) {
  size_t large = static_cast<size_t>(-1) / 2 + 2;
  ASSERT_EQ(LIBC_NAMESPACE::calloc(large, 2), nullptr);
  return 0;
}
