//===-- Tests for random_fill ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/OSUtil/linux/random.h"

#include "test/IntegrationTest/test.h"

void smoke_test() {
  using namespace LIBC_NAMESPACE;
  uint32_t buffer;
  random_fill(&buffer, sizeof(buffer));
}

TEST_MAIN() {
  smoke_test();
  return 0;
}
